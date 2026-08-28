from codexswing.report.idea_html import render_idea_html


def test_report_renders_public_context_and_manual_only_posture() -> None:
    rendered = render_idea_html(
        {
            "as_of_date": "2026-08-27",
            "status": "NO_MANUAL_READY_TRADE",
            "manual_ready_trade_count": 0,
            "raw_orats_underlying_count": 5975,
            "ideas": [],
            "market_context": {
                "status": "SHADOW_ONLY_FALLBACK",
                "provider": "Google News RSS fallback",
                "numeric_vote": False,
                "reason": "display only",
                "items": [
                    {
                        "title": "Source-cited headline",
                        "url": "https://news.example/article",
                        "source_name": "News Example",
                        "published_at_utc": "2026-08-27T12:00:00Z",
                    }
                ],
            },
            "broker_order_authorized": False,
            "broker_order_submitted": False,
            "risk_notice": "research only",
        }
    )
    assert "Public trend and geopolitical context" in rendered
    assert "Source-cited headline" in rendered
    assert "Manual submission only" in rendered
    assert "5,975 ORATS underlyings" in rendered
