import textwrap

import pytest

from uwos.generate_whale_summary import main


def test_generate_whale_summary_rejects_split_part_input(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_whale_summary",
            "--input",
            "bot-eod-report-2026-04-23.part-01-of-03.zip",
        ],
    )

    with pytest.raises(SystemExit, match="split part files are not accepted"):
        main()


def _write_config(path):
    path.write_text(
        textwrap.dedent(
            """
            gates:
              exclude_etfs: true
              exclude_issue_types: ["ETF"]
              min_credit_pct_width: 0.30
              max_credit_pct_width: 0.55
              max_debit_pct_width: 0.45
              min_leg_open_interest: 100
              max_strike_distance_pct: 0.80
              min_whale_premium: 0
              width_tiers:
                - {min_price: 0, max_price: 25, default_width: 2.5}
                - {min_price: 25, max_price: 150, default_width: 10}
            shield:
              dte_range: [28, 56]
              use_anchor_whitelist: false
            fire:
              dte_range: [21, 70]
            """
        ),
        encoding="utf-8",
    )


def test_generate_whale_summary_inferrs_missing_price_from_premium_and_size(tmp_path, monkeypatch):
    csv_path = tmp_path / "bot-eod-report-2026-04-23.csv"
    csv_path.write_text(
        "\n".join(
            [
                "executed_at,underlying_symbol,side,strike,option_type,expiry,underlying_price,price,size,premium,open_interest,equity_type,implied_volatility,delta",
                "2026-04-23 13:30:00+00,AAPL,ask,105,call,2026-05-15,100,,10,2000,500,Common Stock,0.2,0.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "rulebook.yaml"
    output_path = tmp_path / "whale.md"
    _write_config(config_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_whale_summary",
            "--input",
            str(csv_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    text = output_path.read_text(encoding="utf-8")
    assert "Yes-Prime candidates: 1 (100.00%)" in text
    assert "AAPL" in text
    assert "|       2 |      10 |         0.2 |     10 |      2000 |" in text


def test_generate_whale_summary_inferrs_missing_premium_from_price_and_size(tmp_path, monkeypatch):
    csv_path = tmp_path / "bot-eod-report-2026-04-23.csv"
    csv_path.write_text(
        "\n".join(
            [
                "executed_at,underlying_symbol,side,strike,option_type,expiry,underlying_price,price,size,premium,open_interest,equity_type,implied_volatility,delta",
                "2026-04-23 13:30:00+00,AAPL,ask,105,call,2026-05-15,100,2,10,,500,Common Stock,0.2,0.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "rulebook.yaml"
    output_path = tmp_path / "whale.md"
    _write_config(config_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_whale_summary",
            "--input",
            str(csv_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    text = output_path.read_text(encoding="utf-8")
    assert "Yes-Prime candidates: 1 (100.00%)" in text
    assert "AAPL" in text
    assert "2000" in text


def test_generate_whale_summary_uses_report_date_when_executed_at_missing(tmp_path, monkeypatch):
    csv_path = tmp_path / "bot-eod-report-2026-04-23.csv"
    csv_path.write_text(
        "\n".join(
            [
                "executed_at,underlying_symbol,side,strike,option_type,expiry,underlying_price,price,size,premium,open_interest,equity_type,implied_volatility,delta",
                ",AAPL,ask,105,call,2026-05-15,100,2,10,2000,500,Common Stock,0.2,0.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "rulebook.yaml"
    output_path = tmp_path / "whale.md"
    _write_config(config_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_whale_summary",
            "--input",
            str(csv_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    text = output_path.read_text(encoding="utf-8")
    assert "Yes-Prime candidates: 1 (100.00%)" in text
    assert "|    22 |" in text


def test_generate_whale_summary_surfaces_high_premium_rejected_trades(tmp_path, monkeypatch):
    csv_path = tmp_path / "bot-eod-report-2026-04-23.csv"
    csv_path.write_text(
        "\n".join(
            [
                "executed_at,underlying_symbol,side,strike,option_type,expiry,underlying_price,price,size,premium,open_interest,equity_type,implied_volatility,delta",
                "2026-04-23 13:30:00+00,AAPL,ask,105,call,2026-05-15,100,2,10,2000,500,Common Stock,0.2,0.3",
                "2026-04-23 13:30:00+00,NVDA,ask,105,call,2026-04-26,100,2,100,20000,500,Common Stock,0.2,0.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "rulebook.yaml"
    output_path = tmp_path / "whale.md"
    _write_config(config_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_whale_summary",
            "--input",
            str(csv_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    text = output_path.read_text(encoding="utf-8")
    assert "## Top 200 Rejected Trades by Premium (Audit)" in text
    assert "## Top 500 Raw Trades by Premium (Before Filters)" in text
    assert "## Top 500 Yes-Prime Trades by Premium (Extended Audit)" in text
    assert "## Top 500 Rejected Trades by Premium (Extended Audit)" in text
    assert "NVDA" in text
    assert "outside track dte range" in text
    assert "## Raw Top Symbols by Total Premium (Before Filters)" in text


def test_generate_whale_summary_preserves_source_metadata_and_rejects_canceled(tmp_path, monkeypatch):
    csv_path = tmp_path / "bot-eod-report-2026-04-23.csv"
    csv_path.write_text(
        "\n".join(
            [
                "executed_at,underlying_symbol,option_chain_id,side,strike,option_type,expiry,underlying_price,nbbo_bid,nbbo_ask,price,size,premium,volume,open_interest,implied_volatility,delta,sector,exchange,report_flags,canceled,upstream_condition_detail,equity_type",
                "2026-04-23 13:30:00+00,AAPL,AAPL260515C00105000,ask,105,call,2026-05-15,100,1.9,2.1,2,10,2000,100,500,0.2,0.3,Technology,XNAS,{sweep},f,auto,Common Stock",
                "2026-04-23 13:30:00+00,MSFT,MSFT260515C00105000,ask,105,call,2026-05-15,100,1.9,2.1,2,100,20000,1000,500,0.2,0.3,Technology,XNAS,{sweep},t,auto,Common Stock",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "rulebook.yaml"
    output_path = tmp_path / "whale.md"
    _write_config(config_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_whale_summary",
            "--input",
            str(csv_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    text = output_path.read_text(encoding="utf-8")
    assert "Yes-Prime candidates: 1 (50.00%)" in text
    assert "option_chain_id" in text
    assert "nbbo_bid" in text
    assert "volume" in text
    assert "exchange" in text
    assert "report_flags" in text
    assert "AAPL260515C00105000" in text
    assert "MSFT260515C00105000" in text
    assert "canceled trade" in text


def test_generate_whale_summary_rejects_mid_and_no_side_rows(tmp_path, monkeypatch):
    csv_path = tmp_path / "bot-eod-report-2026-04-23.csv"
    csv_path.write_text(
        "\n".join(
            [
                "executed_at,underlying_symbol,side,strike,option_type,expiry,underlying_price,price,size,premium,open_interest,equity_type,implied_volatility,delta",
                "2026-04-23 13:30:00+00,AAPL,ask,105,call,2026-05-15,100,2,10,2000,500,Common Stock,0.2,0.3",
                "2026-04-23 13:30:00+00,MSFT,mid,105,call,2026-05-15,100,2,100,20000,500,Common Stock,0.2,0.3",
                "2026-04-23 13:30:00+00,NVDA,,105,call,2026-05-15,100,2,100,20000,500,Common Stock,0.2,0.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "rulebook.yaml"
    output_path = tmp_path / "whale.md"
    _write_config(config_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_whale_summary",
            "--input",
            str(csv_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    text = output_path.read_text(encoding="utf-8")
    assert "Yes-Prime candidates: 1 (33.33%)" in text
    assert "unsupported side" in text
    assert "UNKNOWN" in text
    assert "MSFT" in text
    assert "NVDA" in text
