"""Stage 6 + 7: live chain pull, candidate selection, and the daily report.

Chains are snapshotted to disk on every run so a day can be re-priced exactly as
it was seen; a live-data pipeline has no valid comparison without that.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from claude_pipeline import PIPELINE_VERSION
from claude_pipeline import panel as panel_mod, context
from claude_pipeline.schwab import SchwabClient

OUT = Path("/Users/anuppamvi/tradedesk/out/claude_pipeline")

# No selection rule has passed validation yet - see research/walkforward.py. The
# credit/width rule that looked strong was an artifact of stale fallback quotes and
# died on true NBBO, so nothing may be presented as executable.
VALIDATED_EDGE = False
EDGE_NOTE = (
    "credit/width >= 0.35 passed walk-forward on blended quotes (PF 1.77) but collapsed "
    "to PF 0.96 when re-priced on closing NBBO only; treated as unvalidated"
)

MIN_CREDIT_PCT_WIDTH = 0.35
DTE_BAND = (21, 45)
MIN_OPEN_INTEREST = 50
MAX_RELATIVE_QUOTE_WIDTH = 0.50
MIN_DOLLAR_VOLUME = 5e7
UNIVERSE_SIZE = 60


@dataclass
class Ticket:
    ticker: str
    family: str
    short_strike: float
    long_strike: float
    expiry: str
    dte: int
    credit: float
    width: float
    credit_pct_width: float
    max_profit: float
    max_loss: float
    return_on_risk: float
    contracts: int
    spot: float
    short_oi: float
    quote_width_pct: float
    status: str
    blocker: str = ""


def build_universe(panel: pd.DataFrame, size: int = UNIVERSE_SIZE) -> list[str]:
    latest = panel["session"].max()
    today = panel[(panel["session"] == latest) & panel["is_equity"]]
    today = today[(today["dollar_volume"] >= MIN_DOLLAR_VOLUME) & (today["close"] > 10)]
    return today.nlargest(size, "dollar_volume")["ticker"].tolist()


def _chain_frame(payload: dict) -> pd.DataFrame:
    rows = []
    for side, key in (("C", "callExpDateMap"), ("P", "putExpDateMap")):
        for expiry_key, strikes in (payload.get(key) or {}).items():
            expiry = expiry_key.split(":")[0]
            for strike, contracts in strikes.items():
                contract = contracts[0]
                rows.append({
                    "kind": side, "expiry": expiry, "strike": float(strike),
                    "bid": contract.get("bid"), "ask": contract.get("ask"),
                    "open_interest": contract.get("openInterest"),
                    "volume": contract.get("totalVolume"),
                    "delta": contract.get("delta"), "iv": contract.get("volatility"),
                    "symbol": contract.get("symbol"),
                })
    return pd.DataFrame(rows)


def _verticals(chain: pd.DataFrame, spot: float, ticker: str, asof: pd.Timestamp) -> list[Ticket]:
    tickets: list[Ticket] = []
    chain = chain.dropna(subset=["bid", "ask"])
    chain = chain[(chain["bid"] > 0) & (chain["ask"] > chain["bid"])]
    if chain.empty:
        return tickets
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    chain["rel_width"] = (chain["ask"] - chain["bid"]) / chain["mid"]
    chain["dte"] = (pd.to_datetime(chain["expiry"]) - asof).dt.days
    chain = chain[(chain["dte"] >= DTE_BAND[0]) & (chain["dte"] <= DTE_BAND[1])]

    for (expiry, kind), side in chain.groupby(["expiry", "kind"]):
        side = side.sort_values("strike")
        family = "bear_call_credit" if kind == "C" else "bull_put_credit"
        candidates = side[side["strike"] >= spot] if kind == "C" else side[side["strike"] <= spot]
        if len(candidates) < 2:
            continue
        rows = list(candidates.itertuples())
        pairs = zip(rows, rows[1:]) if kind == "C" else zip(rows[::-1], rows[-2::-1])
        for short, long_ in pairs:
            width = abs(short.strike - long_.strike)
            if width <= 0:
                continue
            credit = short.bid - long_.ask  # conservative: sell the bid, pay the ask
            ratio = credit / width
            if credit <= 0:
                continue
            max_loss = (width - credit) * 100
            blockers = []
            if ratio < MIN_CREDIT_PCT_WIDTH:
                blockers.append(f"credit {ratio:.0%} of width below validated {MIN_CREDIT_PCT_WIDTH:.0%}")
            if (short.open_interest or 0) < MIN_OPEN_INTEREST:
                blockers.append(f"short-leg OI {short.open_interest:.0f} below {MIN_OPEN_INTEREST}")
            if short.rel_width > MAX_RELATIVE_QUOTE_WIDTH:
                blockers.append(f"quote {short.rel_width:.0%} wide")
            tickets.append(Ticket(
                ticker=ticker, family=family, short_strike=short.strike, long_strike=long_.strike,
                expiry=expiry, dte=int(short.dte), credit=round(credit, 2), width=width,
                credit_pct_width=round(ratio, 3), max_profit=round(credit * 100, 2),
                max_loss=round(max_loss, 2),
                return_on_risk=round(credit * 100 / max_loss, 3) if max_loss > 0 else np.nan,
                contracts=0, spot=spot, short_oi=short.open_interest or 0,
                quote_width_pct=round(short.rel_width, 3),
                status="blocked" if blockers else ("executable" if VALIDATED_EDGE else "watch"),
                blocker="; ".join(blockers) if blockers else ("" if VALIDATED_EDGE else EDGE_NOTE),
            ))
    return tickets


def run(risk_budget: float = 5000.0, max_per_trade: float = 1000.0,
        universe_size: int = UNIVERSE_SIZE) -> pd.DataFrame:
    panel = panel_mod.build()
    regime = context.market_regime(panel)
    client = SchwabClient()
    asof = pd.Timestamp(datetime.now(timezone.utc).date())
    stamp = asof.strftime("%Y-%m-%d")

    snapshot_dir = OUT / "snapshots" / stamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    tickets: list[Ticket] = []
    universe = build_universe(panel, universe_size)
    for ticker in universe:
        try:
            payload = client.option_chain(ticker, strike_count=24)
        except Exception as exc:  # a single bad symbol must not end the run
            print(f"  {ticker}: chain pull failed ({type(exc).__name__})")
            continue
        (snapshot_dir / f"{ticker}.json").write_text(json.dumps(payload))
        spot = payload.get("underlyingPrice")
        if not spot:
            continue
        tickets.extend(_verticals(_chain_frame(payload), float(spot), ticker, asof))

    frame = pd.DataFrame([t.__dict__ for t in tickets])
    if frame.empty:
        print("no candidate structures returned by the live chains")
        return frame

    frame = frame.sort_values(["status", "credit_pct_width"], ascending=[True, False])
    executable = frame[frame["status"] == "executable"].copy()
    if not executable.empty:
        executable["contracts"] = np.minimum(
            np.floor(max_per_trade / executable["max_loss"].clip(lower=1)),
            np.floor(risk_budget / executable["max_loss"].clip(lower=1)),
        ).clip(lower=0).astype(int)
        # one ticket per ticker: same-name spreads are the same bet
        executable = executable.sort_values("credit_pct_width", ascending=False)
        executable = executable.drop_duplicates("ticker", keep="first")
        frame = pd.concat([executable, frame[frame["status"] != "executable"]], ignore_index=True)

    latest_regime = regime.iloc[-1]
    frame["pipeline_version"] = PIPELINE_VERSION
    frame["asof"] = stamp
    frame["market_trend"] = latest_regime["trend"]
    frame["vix"] = latest_regime["vix"]
    frame.to_csv(OUT / f"daily_{stamp}.csv.gz", index=False, compression="gzip")
    _write_report(frame, stamp, latest_regime, len(universe))
    return frame


def _write_report(frame: pd.DataFrame, stamp: str, regime: pd.Series, universe: int) -> None:
    green = frame[(frame["status"] == "executable") & (frame["contracts"] > 0)]
    watch = frame[frame["status"] == "watch"]
    blocked = frame[frame["status"] == "blocked"]

    lines = [f"# Claude Pipeline — {stamp}", "", f"Version: `{PIPELINE_VERSION}`", ""]
    if not VALIDATED_EDGE:
        lines += [
            "**Verdict: no trade today — no validated edge.**",
            "",
            f"The daily scan runs and prices live chains, but {EDGE_NOTE}. "
            "Nothing is presented as executable until a rule passes walk-forward on "
            "closing NBBO. Candidates below are research only.",
        ]
    elif green.empty:
        lines.append("**Verdict: no trade today.** No live structure clears the validated credit floor.")
    else:
        lines.append(
            f"**Verdict: {len(green)} executable.** Median credit "
            f"{green['credit_pct_width'].median():.0%} of width."
        )
    lines += [
        "",
        f"Market: {regime['trend']} | VIX {regime['vix']:.2f} | "
        f"SPX 21d {regime['spx_return_21d']:+.2%} | universe {universe} names",
        "",
        "## Executable",
        "",
        "| Ticker | Structure | Legs | Expiry | DTE | Credit | Width | Cr/W | Max profit | Max loss | RoR | Qty |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in green.itertuples():
        lines.append(
            f"| {row.ticker} | {row.family.replace('_', ' ')} | "
            f"{row.short_strike:g}/{row.long_strike:g} | {row.expiry} | {row.dte} | "
            f"${row.credit:.2f} | ${row.width:g} | {row.credit_pct_width:.0%} | "
            f"${row.max_profit:.0f} | ${row.max_loss:.0f} | {row.return_on_risk:.0%} | {row.contracts} |"
        )
    if green.empty:
        lines.append("| _none_ | | | | | | | | | | | |")

    lines += [
        "", f"## Watch — structurally best available ({len(watch)} candidates, not validated)", "",
        "| Ticker | Structure | Legs | Expiry | DTE | Credit | Cr/W | Max loss | RoR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in watch.nlargest(10, "credit_pct_width").itertuples():
        lines.append(
            f"| {row.ticker} | {row.family.replace('_', ' ')} | "
            f"{row.short_strike:g}/{row.long_strike:g} | {row.expiry} | {row.dte} | "
            f"${row.credit:.2f} | {row.credit_pct_width:.0%} | ${row.max_loss:.0f} | "
            f"{row.return_on_risk:.0%} |"
        )

    lines += ["", "## Blocked", "", "| Count | Reason |", "|---|---|"]
    for reason, count in blocked["blocker"].str.split(";").str[0].value_counts().head(8).items():
        lines.append(f"| {count} | {reason.strip()} |")

    path = OUT / f"report_{stamp}.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"report: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-budget", type=float, default=5000.0)
    parser.add_argument("--max-per-trade", type=float, default=1000.0)
    parser.add_argument("--universe", type=int, default=UNIVERSE_SIZE)
    args = parser.parse_args()
    run(args.risk_budget, args.max_per_trade, args.universe)


if __name__ == "__main__":
    main()
