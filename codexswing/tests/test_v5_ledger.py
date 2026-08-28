import json
import stat

import pytest

from codexswing.v5.ledger import LedgerEvent, ProspectiveLedger, ShadowLedgerRecorder


SPEC_SHA = "a" * 64


def _ledger(tmp_path):
    return ProspectiveLedger(tmp_path / "shadow.jsonl", "codexswing-v0.5", SPEC_SHA)


def test_ledger_is_append_only_hash_chained_idempotent_and_private(tmp_path):
    ledger = _ledger(tmp_path)
    recorder = ShadowLedgerRecorder(ledger)
    first = recorder.signal(
        "SPY-LONG_CALL-1",
        "2026-01-02T21:00:00Z",
        {"status": "HYPOTHETICAL", "limit": 1.25},
    )
    repeated = recorder.signal(
        "SPY-LONG_CALL-1",
        "2026-01-02T21:00:00Z",
        {"status": "HYPOTHETICAL", "limit": 1.25},
    )
    second = recorder.outcome(
        "SPY-LONG_CALL-1",
        "2026-01-09T21:00:00Z",
        {"net_pnl": 20.0, "order_submitted": False},
    )

    assert first == repeated
    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert second["previous_hash"] == first["record_hash"]
    assert ledger.verify().valid
    assert ledger.verify().record_count == 2
    assert stat.S_IMODE(ledger.path.stat().st_mode) == 0o600


def test_ledger_detects_tampering_and_refuses_further_append(tmp_path):
    ledger = _ledger(tmp_path)
    recorder = ShadowLedgerRecorder(ledger)
    recorder.signal("CANDIDATE", "2026-01-02T21:00:00Z", {"limit": 1.0})
    record = json.loads(ledger.path.read_text(encoding="utf-8"))
    record["payload"]["limit"] = 9.0
    ledger.path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert not ledger.verify().valid
    with pytest.raises(ValueError, match="invalid ledger"):
        recorder.outcome("CANDIDATE", "2026-01-09T21:00:00Z", {"net_pnl": 1.0})


def test_ledger_rejects_sensitive_keys(tmp_path):
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError, match="sensitive key"):
        ledger.append(
            LedgerEvent(
                event_id="event-1",
                event_type="SIGNAL",
                occurred_at_utc="2026-01-02T21:00:00Z",
                model_version="codexswing-v0.5",
                spec_sha256=SPEC_SHA,
                candidate_id="CANDIDATE",
                payload={"api_token": "do-not-store"},
            )
        )

