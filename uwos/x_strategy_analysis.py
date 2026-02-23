#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

TRADE_KEYWORDS = [
    'option','options','call','calls','put','puts','spread','credit','debit','iron condor','wheel','theta','premium','naked','covered call','roll','rolled','bullish','bearish'
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Analyze X scraped posts for strategy signals.')
    p.add_argument('--date', required=True)
    p.add_argument('--base-dir', default='C:\\uw_root')
    p.add_argument('--handle', required=True)
    p.add_argument('--scrape-dir', default='AUTO')
    p.add_argument('--rulebook-config', default='')
    p.add_argument('--eod-dir', default='')
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def expand_tokens(value: str, date_text: str) -> str:
    return str(value).replace('{DATE}', date_text).replace('{date}', date_text)


def resolve_scrape_dir(base_dir: Path, date_text: str, handle: str, scrape_dir: str) -> Path:
    if str(scrape_dir).strip().upper() != 'AUTO':
        return Path(scrape_dir).resolve()
    primary = base_dir / date_text / 'x_scrapes' / handle
    if (primary / 'posts.csv').exists():
        return primary
    candidates: List[Tuple[float, Path]] = []
    for p in base_dir.glob(f'*/x_scrapes/{handle}'):
        if (p / 'posts.csv').exists():
            candidates.append(((p / 'posts.csv').stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError(f'Could not resolve scrape folder for handle={handle}')
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def detect_eod_files(eod_dir: Path, date_text: str) -> Dict[str, Optional[Path]]:
    names = {
        'chain_oi': f'chain-oi-changes-{date_text}',
        'dp_eod': f'dp-eod-report-{date_text}',
        'hot_chains': f'hot-chains-{date_text}',
        'stock_screener': f'stock-screener-{date_text}',
        'whale': f'whale-{date_text}',
    }
    out: Dict[str, Optional[Path]] = {k: None for k in names}
    if not eod_dir.exists():
        return out
    for f in eod_dir.rglob('*'):
        if not f.is_file():
            continue
        n = f.name.lower()
        for k, pref in names.items():
            if out[k] is None and n.startswith(pref.lower()) and (n.endswith('.csv') or n.endswith('.zip') or n.endswith('.md')):
                out[k] = f
    return out


def load_rulebook(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not path or not p.exists():
        return {}
    with p.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def text_has_trade_keywords(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in TRADE_KEYWORDS)


def extract_tickers(text: str) -> List[str]:
    vals = re.findall(r'\$([A-Z]{1,6})\b', text.upper())
    out: List[str] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def detect_strategy(text: str) -> str:
    t = text.lower()
    if 'iron condor' in t:
        return 'iron_condor'
    if 'bull call debit' in t or 'call debit spread' in t:
        return 'bull_call_debit'
    if 'bear put debit' in t or 'put debit spread' in t:
        return 'bear_put_debit'
    if 'bull put credit' in t or 'put credit spread' in t:
        return 'bull_put_credit'
    if 'bear call credit' in t or 'call credit spread' in t:
        return 'bear_call_credit'
    return 'other'


def detect_direction(text: str) -> str:
    t = text.lower()
    bull = sum(1 for w in ['bull','bullish','moon','up','rally','call','calls'] if w in t)
    bear = sum(1 for w in ['bear','bearish','bleed','down','crash','put','puts','short'] if w in t)
    if bull > bear:
        return 'bull'
    if bear > bull:
        return 'bear'
    return 'neutral'


def extract_strikes(text: str) -> str:
    legs = re.findall(r'\b(\d+(?:\.\d+)?)\s*([CP])\b', text.upper())
    if legs:
        return ' / '.join([f'{a}{b}' for a, b in legs[:4]])
    pair = re.search(r'\b(\d+(?:\.\d+)?)\s*[-/]\s*(\d+(?:\.\d+)?)\b', text.upper())
    if pair:
        return f"{pair.group(1)} / {pair.group(2)}"
    return 'UNKNOWN'


def extract_expiry(text: str) -> str:
    m = re.search(r'\b(20\d{2}-\d{2}-\d{2})\b', text)
    if m:
        return m.group(1)
    m = re.search(r'\b(\d{1,2}/\d{1,2}/20\d{2})\b', text)
    if m:
        return m.group(1)
    m = re.search(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{1,2}(?:,\s*20\d{2})?\b', text.upper())
    return m.group(0) if m else 'UNKNOWN'


def extract_condition(text: str, kind: str) -> str:
    patterns = {
        'entry': r'(entry|enter|opened|bought|sold)\s+(?:at|around)?\s*\$?\s*(\d+(?:\.\d+)?)',
        'invalidation': r'(stop|invalidat\w*|cut)\s+(?:at|if)?\s*\$?\s*(\d+(?:\.\d+)?)',
        'target': r'(target|tp|take profit|pt)\s+(?:at|near)?\s*\$?\s*(\d+(?:\.\d+)?)',
    }
    m = re.search(patterns[kind], text, flags=re.I)
    if not m:
        return 'UNKNOWN'
    return f"{m.group(1).lower()} near {m.group(2)}"

def scores(ticker: str, strategy: str, strikes: str, expiry: str, entry: str, invalidation: str, target: str, text: str, has_media: bool) -> Tuple[int,int,int,int]:
    comp = 0
    comp += 20 if ticker != 'UNKNOWN' else 0
    comp += 20 if strategy != 'other' else 5
    comp += 15 if strikes != 'UNKNOWN' else 0
    comp += 15 if expiry != 'UNKNOWN' else 0
    comp += 10 if entry != 'UNKNOWN' else 0
    comp += 10 if invalidation != 'UNKNOWN' else 0
    comp += 10 if target != 'UNKNOWN' else 0

    clarity = 20
    if len(text.strip()) >= 30:
        clarity += 20
    if len(text.strip()) >= 80:
        clarity += 20
    if re.search(r'\d', text):
        clarity += 15
    if re.search(r'\$[A-Z]{1,6}\b', text):
        clarity += 10
    clarity = min(100, clarity)

    evidence = 50 + (20 if has_media else 0) + (20 if text.strip() else 0) + (10 if 'http' in text.lower() else 0)
    evidence = min(100, evidence)

    conf = int(round(comp * 0.5 + clarity * 0.2 + evidence * 0.3))
    return comp, clarity, evidence, conf


def readiness(ticker: str, strategy: str, strikes: str, expiry: str, entry: str) -> str:
    if ticker != 'UNKNOWN' and strategy in {'bull_call_debit','bear_put_debit','bull_put_credit','bear_call_credit','iron_condor'} and strikes != 'UNKNOWN' and expiry != 'UNKNOWN' and entry != 'UNKNOWN':
        return 'Ready'
    if ticker != 'UNKNOWN':
        return 'Watch'
    return 'Not actionable'


def gate_check(row: pd.Series, eod_found: Dict[str, Optional[Path]]) -> Tuple[str,str]:
    required = row['ticker'] != 'UNKNOWN' and row['strategy_type'] in {'bull_call_debit','bear_put_debit','bull_put_credit','bear_call_credit','iron_condor'} and row['strikes'] != 'UNKNOWN' and row['expiry'] != 'UNKNOWN'
    if not required:
        return 'FAIL', 'Missing required structure fields (ticker/strategy/strikes/expiry).'
    missing = [k for k,v in eod_found.items() if v is None]
    if missing:
        return 'FAIL', 'Missing EOD files: ' + ', '.join(missing)
    return 'PASS', 'Structure fields present and EOD file set available.'


def to_md(path: Path, title: str, df: pd.DataFrame) -> None:
    lines = [f'## {title}', '']
    lines.append(df.to_markdown(index=False) if not df.empty else 'INSUFFICIENT DATA')
    path.write_text('\n'.join(lines), encoding='utf-8')


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = 1
    for _, r in df.iterrows():
        text = str(r.get('text', '') or '').strip()
        if not text_has_trade_keywords(text):
            continue
        tickers = extract_tickers(text) or ['UNKNOWN']
        strategy = detect_strategy(text)
        direction = detect_direction(text)
        strikes = extract_strikes(text)
        expiry = extract_expiry(text)
        entry = extract_condition(text, 'entry')
        invalidation = extract_condition(text, 'invalidation')
        target = extract_condition(text, 'target')
        snippet = text.replace('\n', ' ')[:220] + ('...' if len(text) > 220 else '')

        for t in tickers:
            comp, clarity, evid, conf = scores(t, strategy, strikes, expiry, entry, invalidation, target, text, bool(r.get('has_media', False)))
            rows.append({
                'signal_id': f'S{n:03d}',
                'tweet_id': str(r.get('tweet_id', '') or ''),
                'tweet_url': str(r.get('tweet_url', '') or ''),
                'published_at': str(r.get('published_at', '') or ''),
                'source_file': str(r.get('source_file', '') or 'posts.csv'),
                'ticker': t,
                'direction': direction,
                'strategy_type': strategy,
                'strikes': strikes,
                'expiry': expiry,
                'entry_condition': entry,
                'invalidation': invalidation,
                'target': target,
                'thesis': 'Derived from post text; hypothesis until structure/pricing explicit.',
                'fact_summary': f'FACT: {snippet}',
                'inference_summary': 'INFERENCE: strategy inferred from language only.' if strategy == 'other' else 'INFERENCE: directional/strategy mapping uses keyword heuristics.',
                'completeness_score': comp,
                'clarity_score': clarity,
                'evidence_score': evid,
                'confidence_score': conf,
                'confidence_bucket': 'High' if conf >= 75 else ('Medium' if conf >= 50 else 'Low'),
                'execution_readiness': readiness(t, strategy, strikes, expiry, entry),
                'has_media': bool(r.get('has_media', False)),
                'has_text': bool(text),
                'evidence_snippet': snippet,
            })
            n += 1

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out['published_at_ts'] = pd.to_datetime(out['published_at'], errors='coerce', utc=True)
    out = out.sort_values(['published_at_ts', 'confidence_score'], ascending=[False, False]).reset_index(drop=True)
    return out

def main() -> int:
    a = parse_args()
    date_text = str(a.date).strip()
    base_dir = Path(expand_tokens(a.base_dir, date_text)).resolve()
    out_dir = Path(expand_tokens(a.out_dir, date_text)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eod_dir = Path(expand_tokens(a.eod_dir if a.eod_dir else str(base_dir / date_text), date_text)).resolve()
    scrape_dir = resolve_scrape_dir(base_dir, date_text, a.handle, expand_tokens(a.scrape_dir, date_text))

    posts_csv = scrape_dir / 'posts.csv'
    posts_jsonl = scrape_dir / 'posts.jsonl'
    if not posts_csv.exists() or not posts_jsonl.exists():
        raise FileNotFoundError('Required files missing: posts.csv and/or posts.jsonl')

    rb = load_rulebook(expand_tokens(a.rulebook_config, date_text))
    eod_found = detect_eod_files(eod_dir, date_text)

    df = pd.read_csv(posts_csv, low_memory=False)
    if 'source_file' not in df.columns:
        df['source_file'] = 'posts.csv'
    df['published_at_ts'] = pd.to_datetime(df.get('published_at', ''), errors='coerce', utc=True)
    df = df.sort_values('published_at_ts', ascending=False).reset_index(drop=True)

    raw = build_signals(df)
    if raw.empty:
        sig = raw
    else:
        latest = []
        for _, g in raw.sort_values('published_at_ts', ascending=False).groupby(['ticker','direction','strategy_type'], dropna=False):
            r = g.iloc[0].to_dict()
            ts = g['published_at_ts'].dropna()
            first_d = ts.min().date().isoformat() if len(ts) else 'UNKNOWN'
            last_d = ts.max().date().isoformat() if len(ts) else 'UNKNOWN'
            pri = [str(x) for x in g['tweet_id'].astype(str).tolist()[1:4]]
            r['related_post_count'] = int(len(g))
            r['change_log'] = f"{len(g)} related posts, window {first_d} to {last_d}; prior tweet_ids={','.join(pri) if pri else 'none'}"
            latest.append(r)
        sig = pd.DataFrame(latest).sort_values('confidence_score', ascending=False).reset_index(drop=True)
        sig['rank'] = sig.index + 1

    strategies = sorted(sig['strategy_type'].dropna().astype(str).unique().tolist()) if not sig.empty else []
    debit_cap = rb.get('gates', {}).get('max_debit_pct_width', 'UNKNOWN')
    credit_floor = rb.get('gates', {}).get('min_credit_pct_width', 'UNKNOWN')
    fire_dte = rb.get('fire', {}).get('dte_range', 'UNKNOWN')
    shield_dte = rb.get('shield', {}).get('dte_range', 'UNKNOWN')

    trows = []
    for i, s in enumerate(strategies, start=1):
        if s in {'bull_call_debit', 'bear_put_debit'}:
            trows.append({'Template ID': f'T{i:02d}', 'Strategy': s, 'Market Regime': 'Directional momentum', 'Entry Rule': f'debit <= {debit_cap} width', 'Exit Rule': 'close-confirm invalidation + profit target', 'Risk Rule': f'FIRE DTE {fire_dte}', 'Minimum Data Needed': 'ticker,strikes,expiry,live bid/ask'})
        elif s in {'bull_put_credit', 'bear_call_credit', 'iron_condor'}:
            trows.append({'Template ID': f'T{i:02d}', 'Strategy': s, 'Market Regime': 'Income / mean reversion', 'Entry Rule': f'credit >= {credit_floor} width', 'Exit Rule': 'capture profit; roll if threatened', 'Risk Rule': f'SHIELD DTE {shield_dte}', 'Minimum Data Needed': 'ticker,strikes,expiry,live bid/ask,earnings'})
        else:
            trows.append({'Template ID': f'T{i:02d}', 'Strategy': s, 'Market Regime': 'Unspecified', 'Entry Rule': 'convert to defined-risk spread', 'Exit Rule': 'pre-define before entry', 'Risk Rule': 'no naked risk without explicit plan', 'Minimum Data Needed': 'explicit setup details'})
    templates = pd.DataFrame(trows)

    tfreq = sig.loc[sig['ticker'] != 'UNKNOWN', 'ticker'].value_counts().to_dict() if not sig.empty else {}
    tickers = list(tfreq.keys())
    arows = []
    rnk = 1
    if not sig.empty:
        for _, s in sig.iterrows():
            g, reason = gate_check(s, eod_found)
            act = 'Ready' if g == 'PASS' and s['execution_readiness'] == 'Ready' else ('Reject' if s['ticker'] == 'UNKNOWN' else 'Watch')
            arows.append({'#': rnk, 'Source Signal ID': s['signal_id'], 'Candidate Ticker': s['ticker'], 'Same/Alt': 'Same', 'Proposed Setup': s['strategy_type'], 'Gate Check': f'{g} - {reason}', 'Action': act, 'Why': 'Direct carry-over from source signal.', 'Evidence': s['tweet_url']})
            rnk += 1
            alt = next((t for t in tickers if t != s['ticker']), None)
            if alt:
                arows.append({'#': rnk, 'Source Signal ID': s['signal_id'], 'Candidate Ticker': alt, 'Same/Alt': 'Alt', 'Proposed Setup': s['strategy_type'], 'Gate Check': 'FAIL - Transfer candidate only; no ticker-specific structure/pricing in source.', 'Action': 'Watch' if s['ticker'] != 'UNKNOWN' else 'Reject', 'Why': 'Same strategy logic, different ticker from profile mentions.', 'Evidence': s['tweet_url']})
                rnk += 1
    apps = pd.DataFrame(arows)

    signal_csv = out_dir / 'x_signal_extract.csv'
    signal_md = out_dir / 'x_signal_extract.md'
    transfer_csv = out_dir / 'x_strategy_transfer.csv'
    transfer_md = out_dir / 'x_strategy_transfer.md'
    summary_md = out_dir / 'x_analysis_summary.md'

    signal_cols = ['rank','signal_id','ticker','direction','strategy_type','strikes','expiry','entry_condition','invalidation','target','execution_readiness','confidence_score','confidence_bucket','tweet_url','published_at','evidence_snippet','fact_summary','inference_summary','related_post_count','change_log']
    if sig.empty:
        pd.DataFrame(columns=signal_cols).to_csv(signal_csv, index=False)
        signal_view = pd.DataFrame()
    else:
        sig[signal_cols].to_csv(signal_csv, index=False)
        signal_view = sig[['rank','signal_id','ticker','direction','strategy_type','strikes','expiry','entry_condition','invalidation','target','execution_readiness','confidence_score','tweet_url','published_at','evidence_snippet']].rename(columns={'rank':'#','strategy_type':'Strategy Type','entry_condition':'Entry','execution_readiness':'Readiness','confidence_score':'Confidence','tweet_url':'Evidence'})

    apps.to_csv(transfer_csv, index=False)
    to_md(signal_md, 'Extracted Source Signals', signal_view)
    to_md(transfer_md, 'Applications (Same + Alt Tickers)', apps)

    have_ocr = bool(importlib.util.find_spec('PIL')) and bool(importlib.util.find_spec('pytesseract'))
    gaps = []
    for k, v in eod_found.items():
        if v is None:
            gaps.append(f'Missing EOD file for {k}.')
    if not have_ocr:
        gaps.append('OCR stack missing (Pillow/pytesseract/tesseract), so screenshot-only text was not extracted.')
    if not sig.empty:
        u1 = float((sig['strikes'] == 'UNKNOWN').mean()) * 100.0
        u2 = float((sig['expiry'] == 'UNKNOWN').mean()) * 100.0
        if u1 > 40:
            gaps.append(f'High missing strike detail ({u1:.1f}% UNKNOWN).')
        if u2 > 40:
            gaps.append(f'High missing expiry detail ({u2:.1f}% UNKNOWN).')

    files_used = [posts_csv, posts_jsonl, scrape_dir / 'scrape_summary.md'] + [v for v in eod_found.values() if v is not None]
    lines = [f'As-of date used: {date_text}', f'Scrape folder used: {scrape_dir}', 'Files used:']
    lines.extend([f'- {x}' for x in files_used if x.exists()])
    lines += ['', '## Extracted Source Signals', '', (signal_view.to_markdown(index=False) if not signal_view.empty else 'INSUFFICIENT DATA'), '', '## Reusable Strategy Templates', '', (templates.to_markdown(index=False) if not templates.empty else 'INSUFFICIENT DATA'), '', '## Applications (Same + Alt Tickers)', '', (apps.to_markdown(index=False) if not apps.empty else 'INSUFFICIENT DATA'), '', '## Top 5 actionable ideas now', '']

    if sig.empty:
        lines.append('INSUFFICIENT DATA')
    else:
        for i, (_, r) in enumerate(sig.sort_values('confidence_score', ascending=False).head(5).iterrows(), start=1):
            strat = str(r['strategy_type'])
            if strat in {'bull_call_debit', 'bear_put_debit'}:
                entry = f'Only execute if defined spread is available with debit <= {debit_cap} of width at executable quotes.'
            elif strat in {'bull_put_credit', 'bear_call_credit', 'iron_condor'}:
                entry = f'Only execute if credit >= {credit_floor} of width at executable quotes.'
            else:
                entry = 'Convert to a defined-risk spread first; no execution on vague signal.'
            inv = 'Two daily closes against thesis level; define concrete level before entry.'
            lines += [f"{i}. {r['ticker']} ({r['signal_id']})", f"- thesis: {r['ticker']} signal inferred from source post language.", f"- exact entry condition: {entry}", f"- invalidation: {inv}", '- no-trade condition: Missing strikes/expiry/entry pricing or missing EOD validation files.']

    lines += ['', '## Data Gaps', '']
    lines += [f'- {g}' for g in gaps] if gaps else ['- None']
    summary_md.write_text('\n'.join(lines), encoding='utf-8')

    print('\n'.join(lines))
    print('\nSaved outputs:')
    print(f'- {signal_csv}')
    print(f'- {signal_md}')
    print(f'- {transfer_csv}')
    print(f'- {transfer_md}')
    print(f'- {summary_md}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
