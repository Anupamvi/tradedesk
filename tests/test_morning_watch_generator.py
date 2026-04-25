import sys

import pandas as pd

from uwos import generate_chain_only_watchlist as morning


def test_morning_watch_generator_writes_headers_when_no_setups_survive(tmp_path, monkeypatch):
    run_date = "2026-04-23"
    output_csv = tmp_path / "morning-watch-setups-2026-04-23.csv"
    output_md = tmp_path / "morning-watch-setups-2026-04-23.md"
    fake_chain = pd.DataFrame({"ticker": ["ZZZ", "AAA"]})

    monkeypatch.setattr(morning, "_load_chain", lambda *args, **kwargs: fake_chain.copy())
    monkeypatch.setattr(morning, "_ticker_summary", lambda rows: {})
    monkeypatch.setattr(morning, "_rank_ticker_summary", lambda summary: 0.0)

    def fake_build_candidate(*args, **kwargs):
        return None, morning.Exclusion(
            ticker=str(args[1]),
            status="EXCLUDED",
            reason="No deterministic setup survived.",
        )

    monkeypatch.setattr(morning, "_build_candidate_for_ticker", fake_build_candidate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_chain_only_watchlist.py",
            "--date",
            run_date,
            "--base-dir",
            str(tmp_path),
            "--chain-csv",
            str(tmp_path / "chain-oi-changes-2026-04-23.csv"),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
            "--focus-tickers",
            "ZZZ",
            "--historical-replay",
        ],
    )

    assert morning.main() == 0

    generated = pd.read_csv(output_csv)
    assert list(generated.columns) == morning.MORNING_WATCH_COLUMNS
    assert generated.empty
