#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Infer range-game options logic from scraped X posts.')
    p.add_argument('--date', required=True)
    p.add_argument('--base-dir', default='C:\\uw_root')
    p.add_argument('--handle', required=True)
    p.add_argument('--scrape-dir', default='AUTO')
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def resolve_scrape_dir(base_dir: Path, date_text: str, handle: str, scrape_dir: str) -> Path:
    if str(scrape_dir).upper() != 'AUTO':
        return Path(scrape_dir).resolve()
    primary = base_dir / date_text / 'x_scrapes' / handle
    if (primary / 'posts.csv').exists():
        return primary
    cands: List[Path] = []
    for p in base_dir.glob(f'*/x_scrapes/{handle}'):
        if (p / 'posts.csv').exists():
            cands.append(p)
    if not cands:
        raise FileNotFoundError(f'No scrape folder found for {handle}')
    cands.sort(key=lambda x: (x / 'posts.csv').stat().st_mtime, reverse=True)
    return cands[0]


def extract_events(text: str, tweet_id: str, tweet_url: str, published_at: str) -> List[Dict[str, Any]]:
    t = str(text or '')
    out: List[Dict[str, Any]] = []

    # Pattern: agreement to buy/sell shares of $TICKER @$STRIKE XX days from now
    p_agree = re.compile(
        r'agreement to\s+'
        r'(buy|sell)\s+'
        r'([\d,.]+\s*[kKmM]?)?\s*'
        r'(?:shares?|sharas?)\s+of\s+'
        r'\$([A-Z]{1,6})\s*@\$?(\d+(?:\.\d+)?)\s+'
        r'(\d+)\s+days',
        flags=re.I,
    )
    for m in p_agree.finditer(t):
        side = m.group(1).lower()
        qty = m.group(2) or ''
        ticker = m.group(3).upper()
        strike = float(m.group(4))
        dte = int(m.group(5))
        leg = 'short_put_floor' if side == 'buy' else 'short_call_ceiling'
        out.append(
            {
                'tweet_id': tweet_id,
                'tweet_url': tweet_url,
                'published_at': published_at,
                'ticker': ticker,
                'event_type': 'agreement',
                'leg_role': leg,
                'action': side,
                'qty_text': qty,
                'level': strike,
                'dte_days': dte,
                'evidence': m.group(0),
            }
        )

    # Pattern: lowered naked call strikes on $TICKER to $X
    p_lower_call = re.compile(r'lowered\s+naked\s+call\s+strikes\s+on\s+\$([A-Z]{1,6})\s+to\s+\$?(\d+(?:\.\d+)?)', flags=re.I)
    for m in p_lower_call.finditer(t):
        out.append(
            {
                'tweet_id': tweet_id,
                'tweet_url': tweet_url,
                'published_at': published_at,
                'ticker': m.group(1).upper(),
                'event_type': 'adjustment',
                'leg_role': 'short_call_ceiling',
                'action': 'lower_strike',
                'qty_text': '',
                'level': float(m.group(2)),
                'dte_days': None,
                'evidence': m.group(0),
            }
        )

    # Pattern: sold N more naked calls on $TICKER
    p_sold_naked_calls = re.compile(r'sold\s+([\d,]+)?\s*more\s+naked\s+calls\s+on\s+\$([A-Z]{1,6})', flags=re.I)
    for m in p_sold_naked_calls.finditer(t):
        out.append(
            {
                'tweet_id': tweet_id,
                'tweet_url': tweet_url,
                'published_at': published_at,
                'ticker': m.group(2).upper(),
                'event_type': 'position_add',
                'leg_role': 'short_call_ceiling',
                'action': 'sell_calls',
                'qty_text': m.group(1) or '',
                'level': None,
                'dte_days': None,
                'evidence': m.group(0),
            }
        )

    # Pattern: $TICKER below/above $LEVEL
    p_level = re.compile(r'\$([A-Z]{1,6})\s+(below|above)\s+\$?(\d+(?:\.\d+)?)', flags=re.I)
    for m in p_level.finditer(t):
        relation = m.group(2).lower()
        role = 'short_call_ceiling' if relation == 'below' else 'short_put_floor'
        out.append(
            {
                'tweet_id': tweet_id,
                'tweet_url': tweet_url,
                'published_at': published_at,
                'ticker': m.group(1).upper(),
                'event_type': 'range_condition',
                'leg_role': role,
                'action': relation,
                'qty_text': '',
                'level': float(m.group(3)),
                'dte_days': None,
                'evidence': m.group(0),
            }
        )

    # Premium capture mentions
    p_premium = re.compile(r'(COLLECTED|premium haul|capture premiums?|theta collection)[^\n\.]*', flags=re.I)
    for m in p_premium.finditer(t):
        # attach to UNKNOWN ticker unless cashtag exists
        cashtags = re.findall(r'\$([A-Z]{1,6})', t)
        ticker = cashtags[0].upper() if cashtags else 'UNKNOWN'
        out.append(
            {
                'tweet_id': tweet_id,
                'tweet_url': tweet_url,
                'published_at': published_at,
                'ticker': ticker,
                'event_type': 'premium_objective',
                'leg_role': 'portfolio',
                'action': 'premium_first',
                'qty_text': '',
                'level': None,
                'dte_days': None,
                'evidence': m.group(0).strip(),
            }
        )

    return out


def infer_ticker_logic(events: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ticker, g in events.groupby('ticker'):
        call_levels = sorted([float(x) for x in g.loc[g['leg_role'] == 'short_call_ceiling', 'level'].dropna().tolist()])
        put_levels = sorted([float(x) for x in g.loc[g['leg_role'] == 'short_put_floor', 'level'].dropna().tolist()])
        call_dte = [int(x) for x in g.loc[(g['leg_role'] == 'short_call_ceiling') & g['dte_days'].notna(), 'dte_days'].tolist()]
        put_dte = [int(x) for x in g.loc[(g['leg_role'] == 'short_put_floor') & g['dte_days'].notna(), 'dte_days'].tolist()]
        premium_hits = int((g['event_type'] == 'premium_objective').sum())
        adjust_hits = int((g['event_type'] == 'adjustment').sum())

        if call_levels and put_levels:
            style = 'Two-sided range monetization (sell ceiling + own/accept floor)'
        elif call_levels:
            style = 'Ceiling monetization via short calls / call overwrites'
        elif put_levels:
            style = 'Floor acquisition via short puts / backstop buying'
        else:
            style = 'Directional/options commentary'

        logic = []
        if call_levels:
            logic.append(f'Ceiling levels observed: {call_levels}')
        if put_levels:
            logic.append(f'Floor levels observed: {put_levels}')
        if call_dte:
            logic.append(f'Call-side DTE samples: {sorted(call_dte)}')
        if put_dte:
            logic.append(f'Put-side DTE samples: {sorted(put_dte)}')
        if adjust_hits:
            logic.append('Actively adjusts call strikes (roll/lower).')
        if premium_hits:
            logic.append('Premium/theta capture explicitly prioritized.')

        # Replication rules from inferred style
        if 'Two-sided range' in style:
            repl = 'Set a floor you are willing to own; finance via shorter-dated call sales near upper range.'
        elif 'Ceiling monetization' in style:
            repl = 'Use call overwrites or bear call overlays near resistance; adjust lower if spot weakens.'
        elif 'Floor acquisition' in style:
            repl = 'Sell puts only at levels you want to own; use farther DTE for assignment tolerance.'
        else:
            repl = 'Needs explicit strikes/expiry before replication.'

        rows.append(
            {
                'ticker': ticker,
                'posts_count': int(g['tweet_id'].nunique()),
                'events_count': int(len(g)),
                'inferred_style': style,
                'call_ceiling_min': min(call_levels) if call_levels else None,
                'call_ceiling_max': max(call_levels) if call_levels else None,
                'put_floor_min': min(put_levels) if put_levels else None,
                'put_floor_max': max(put_levels) if put_levels else None,
                'call_dte_min': min(call_dte) if call_dte else None,
                'call_dte_max': max(call_dte) if call_dte else None,
                'put_dte_min': min(put_dte) if put_dte else None,
                'put_dte_max': max(put_dte) if put_dte else None,
                'replication_rule': repl,
                'logic_notes': '; '.join(logic) if logic else 'Insufficient structured evidence',
            }
        )
    return pd.DataFrame(rows).sort_values(['events_count', 'posts_count'], ascending=False)


def ticker_status_row(row: pd.Series) -> Dict[str, str]:
    two_sided = bool(pd.notna(row.get('call_ceiling_min')) and pd.notna(row.get('put_floor_min')))
    call_only = bool(pd.notna(row.get('call_ceiling_min')) and pd.isna(row.get('put_floor_min')))
    events = int(row.get('events_count', 0) or 0)
    style = str(row.get('inferred_style', ''))

    if two_sided and events >= 3:
        return {
            'label': '🟢 [GREEN] Valid Framework',
            'color': 'Green',
            'fit': 'Yes (with risk controls)',
            'row_color': '#e8f5e9',
            'guidance': 'Replicate as defined-risk range income; keep floor/ceiling levels and DTE split.',
            'valid_now': 'Conditional Yes',
        }
    if call_only and events >= 2:
        return {
            'label': '🟡 [YELLOW] Partial Framework',
            'color': 'Yellow',
            'fit': 'Partially',
            'row_color': '#fff8e1',
            'guidance': 'You only have ceiling-side logic. Add floor plan (assignment level) before full replication.',
            'valid_now': 'Watch',
        }
    if 'Directional/options commentary' in style or events <= 1:
        return {
            'label': '🔴 [RED] Insufficient Structure',
            'color': 'Red',
            'fit': 'No',
            'row_color': '#ffebee',
            'guidance': 'Do not replicate directly. Need explicit strikes, expiry, and adjustment rule set.',
            'valid_now': 'No',
        }
    return {
        'label': '🟠 [ORANGE] Low Confidence',
        'color': 'Orange',
        'fit': 'Not yet',
        'row_color': '#fff3e0',
        'guidance': 'Keep on watchlist; wait for additional structured posts before acting.',
        'valid_now': 'Watch',
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scrape_dir = resolve_scrape_dir(base_dir, args.date, args.handle, args.scrape_dir)
    posts_csv = scrape_dir / 'posts.csv'
    if not posts_csv.exists():
        raise FileNotFoundError(f'Missing {posts_csv}')

    df = pd.read_csv(posts_csv, low_memory=False)
    df['text'] = df['text'].fillna('').astype(str)

    trade_mask = df['text'].str.contains(r'\$[A-Z]{1,6}|naked call|short put|short call|sell options|theta|premium|roll|agreement to buy|agreement to sell|below|above|puts|calls', case=False, regex=True)
    trade_posts = df[trade_mask].copy()

    events: List[Dict[str, Any]] = []
    for _, r in trade_posts.iterrows():
        events.extend(extract_events(r['text'], str(r.get('tweet_id', '')), str(r.get('tweet_url', '')), str(r.get('published_at', ''))))

    events_df = pd.DataFrame(events)
    if events_df.empty:
        ticker_logic_df = pd.DataFrame(columns=['ticker', 'inferred_style', 'replication_rule', 'logic_notes'])
    else:
        ticker_logic_df = infer_ticker_logic(events_df)

    # Candidate replication universe from mentioned tickers
    mentioned = sorted(set(re.findall(r'\$([A-Z]{1,6})', ' '.join(df['text'].tolist()))))
    repl_rows = []
    for t in mentioned:
        if t in ticker_logic_df['ticker'].tolist():
            src = ticker_logic_df[ticker_logic_df['ticker'] == t].iloc[0]
            repl_rows.append({
                'ticker': t,
                'source_style': src['inferred_style'],
                'replication_candidate': 'Same ticker',
                'how_to_apply': src['replication_rule'],
                'confidence': 'Medium' if src['events_count'] >= 2 else 'Low',
                'notes': src['logic_notes'],
            })
        else:
            repl_rows.append({
                'ticker': t,
                'source_style': 'No structured leg evidence',
                'replication_candidate': 'Watchlist only',
                'how_to_apply': 'Need explicit strike + expiry posts before replication.',
                'confidence': 'Low',
                'notes': 'Mentioned but no range-leg structure extracted.',
            })
    repl_df = pd.DataFrame(repl_rows)

    events_csv = out_dir / 'x_range_events.csv'
    logic_csv = out_dir / 'x_range_ticker_logic.csv'
    repl_csv = out_dir / 'x_range_replication_candidates.csv'
    summary_md = out_dir / 'x_range_strategy_summary.md'

    events_df.to_csv(events_csv, index=False)
    ticker_logic_df.to_csv(logic_csv, index=False)
    repl_df.to_csv(repl_csv, index=False)

    lines: List[str] = []
    lines.append(f'As-of date used: {args.date}')
    lines.append(f'Scrape folder used: {scrape_dir}')
    lines.append('Files used:')
    lines.append(f'- {posts_csv}')
    lines.append('')
    lines.append('## What He Is Doing (Inferred)')
    lines.append('')
    lines.append('- Core objective is premium/theta harvesting, not directional prediction.')
    lines.append('- Uses a range framework: monetize upside ceiling with short calls while tolerating/structuring downside via short puts or buy-agreements at lower levels.')
    lines.append('- Adjusts call strikes downward when market weakens to keep collecting premium.')
    lines.append('- Legend: 🟢 Green = valid framework, 🟡 Yellow = partial/watch, 🔴 Red = not valid yet.')
    lines.append('')

    lines.append('## 1) What He Did Per Ticker')
    lines.append('')
    if ticker_logic_df.empty or events_df.empty:
        lines.append('INSUFFICIENT DATA')
    else:
        detail_rows: List[Dict[str, Any]] = []
        for _, r in ticker_logic_df.iterrows():
            t = str(r['ticker'])
            if t == 'UNKNOWN':
                continue
            ev = events_df[events_df['ticker'] == t].copy()
            ev = ev.sort_values('published_at')
            evidence = []
            for _, er in ev.head(3).iterrows():
                tid = str(er.get('tweet_id', ''))
                evidence.append(f"{tid}:{str(er.get('evidence', ''))[:60]}")
            levels = []
            if pd.notna(r.get('put_floor_min')):
                levels.append(f"Floor {r['put_floor_min']}")
            if pd.notna(r.get('call_ceiling_min')):
                if pd.notna(r.get('call_ceiling_max')) and float(r['call_ceiling_max']) != float(r['call_ceiling_min']):
                    levels.append(f"Ceiling {r['call_ceiling_min']}-{r['call_ceiling_max']}")
                else:
                    levels.append(f"Ceiling {r['call_ceiling_min']}")
            dte = []
            if pd.notna(r.get('put_dte_min')):
                dte.append(f"Put DTE {int(float(r['put_dte_min']))}{('-' + str(int(float(r['put_dte_max'])))) if pd.notna(r.get('put_dte_max')) and int(float(r['put_dte_max'])) != int(float(r['put_dte_min'])) else ''}")
            if pd.notna(r.get('call_dte_min')):
                dte.append(f"Call DTE {int(float(r['call_dte_min']))}{('-' + str(int(float(r['call_dte_max'])))) if pd.notna(r.get('call_dte_max')) and int(float(r['call_dte_max'])) != int(float(r['call_dte_min'])) else ''}")
            detail_rows.append(
                {
                    'Ticker': t,
                    'Style': str(r['inferred_style']),
                    'What He Did': str(r['logic_notes']).replace('|', ';'),
                    'Range Levels': ', '.join(levels) if levels else 'Not explicit',
                    'Timing': ', '.join(dte) if dte else 'Not explicit',
                    'Evidence (tweet_id:snippet)': ' ; '.join(evidence) if evidence else 'N/A',
                }
            )

        detail_df = pd.DataFrame(detail_rows)
        lines.append(detail_df.to_markdown(index=False))

    lines.append('')
    lines.append('## 2) What You Should Do (Is This Valid For You?)')
    lines.append('')
    if ticker_logic_df.empty:
        lines.append('INSUFFICIENT DATA')
    else:
        action_rows: List[Dict[str, Any]] = []
        for _, r in ticker_logic_df.iterrows():
            t = str(r['ticker'])
            if t == 'UNKNOWN':
                continue
            status = ticker_status_row(r)
            levels = []
            if pd.notna(r.get('put_floor_min')):
                levels.append(f"Floor={r['put_floor_min']}")
            if pd.notna(r.get('call_ceiling_min')):
                levels.append(f"Ceiling={r['call_ceiling_min']}")
            risk_guard = 'Use defined-risk alternatives; avoid oversized naked exposure.'
            action_rows.append(
                {
                    'Ticker': t,
                    'Color': status['color'],
                    'Status': status['label'],
                    'Trade Valid For You?': status['valid_now'],
                    'Why': status['guidance'],
                    'How To Apply': str(r['replication_rule']),
                    'Key Levels': ', '.join(levels) if levels else 'Need more data',
                    'Risk Guardrails': risk_guard,
                }
            )

        action_df = pd.DataFrame(action_rows)
        lines.append(action_df.to_markdown(index=False))

    lines.append('')
    lines.append('### BMNR / PLTR Quick Read')
    lines.append('')
    bmnr = ticker_logic_df[ticker_logic_df['ticker'] == 'BMNR']
    pltr = ticker_logic_df[ticker_logic_df['ticker'] == 'PLTR']
    if bmnr.empty:
        lines.append('- BMNR: 🟡 Mentioned as a vehicle for selling options, but this scrape window lacks explicit BMNR strike+DTE pair.')
    else:
        rb = bmnr.iloc[0]
        lines.append(f"- BMNR: {ticker_status_row(rb)['label']} | {rb['logic_notes']}")
    if pltr.empty:
        lines.append('- PLTR: 🔴 Missing in structured extraction (unexpected).')
    else:
        rp = pltr.iloc[0]
        lines.append(f"- PLTR: {ticker_status_row(rp)['label']} | {rp['logic_notes']}")

    lines.append('')
    lines.append('## Data Gaps')
    lines.append('')
    lines.append('- Some posts are text-truncated in card scrape; full-thread expansion/OCR would improve leg extraction.')
    lines.append('- No broker chain snapshot in this run; replication here is style-level, not executable pricing-level.')

    summary_md.write_text('\n'.join(lines), encoding='utf-8-sig')

    print('\n'.join(lines))
    print('')
    print('Saved outputs:')
    print(f'- {events_csv}')
    print(f'- {logic_csv}')
    print(f'- {repl_csv}')
    print(f'- {summary_md}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
