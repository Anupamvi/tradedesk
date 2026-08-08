"""Profile the five Unusual Whales daily files: schema, size, null rates, samples.

Investigation only - no assumptions about what the columns mean.
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

UW_ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FAMILIES = ("stock-screener", "hot-chains", "chain-oi-changes", "bot-eod-report", "dp-eod-report")


def dated_dirs(root: Path = UW_ROOT) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and DATE_DIR.match(d.name))


def family_zips(day: Path, family: str) -> list[Path]:
    """All zips for a family on a day, excluding 'latest-' variants."""
    return sorted(p for p in day.glob(f"{family}-*.zip") if "latest" not in p.name)


def count_rows(zip_path: Path) -> int:
    total = 0
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            if not member.filename.endswith(".csv"):
                continue
            with z.open(member) as handle:
                first = True
                while chunk := handle.read(16 << 20):
                    total += chunk.count(b"\n")
                    if first and not chunk.endswith(b"\n"):
                        pass
                    first = False
            total -= 1  # header
    return total


def sample_frame(zip_path: Path, nrows: int = 20000) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        member = next(m for m in z.infolist() if m.filename.endswith(".csv"))
        with z.open(member) as handle:
            raw = handle.read(64 << 20)
    return pd.read_csv(io.BytesIO(raw), nrows=nrows, low_memory=False)


def profile(day: Path, family: str) -> None:
    zips = family_zips(day, family)
    print(f"\n{'=' * 100}\n{family}  ({day.name})")
    if not zips:
        print("  MISSING")
        return
    print(f"  parts: {len(zips)} -> {[p.name for p in zips]}")
    rows = sum(count_rows(p) for p in zips)
    frame = sample_frame(zips[0])
    print(f"  total rows (all parts): {rows:,}   columns: {len(frame.columns)}   (profiled on first {len(frame):,} rows)")
    print(f"  {'column':<28}{'dtype':<12}{'null%':>7}{'nuniq':>9}  example values")
    for col in frame.columns:
        series = frame[col]
        nulls = series.isna().mean() * 100
        examples = [str(v) for v in series.dropna().unique()[:3]]
        text = ", ".join(examples)[:52]
        print(f"  {col:<28}{str(series.dtype):<12}{nulls:>6.1f}%{series.nunique():>9}  {text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD; defaults to latest folder")
    parser.add_argument("--family", default=None, help="profile only one family")
    args = parser.parse_args()

    days = dated_dirs()
    day = next(d for d in reversed(days) if d.name == args.date) if args.date else days[-1]
    families = [args.family] if args.family else list(FAMILIES)
    for family in families:
        profile(day, family)


if __name__ == "__main__":
    main()
