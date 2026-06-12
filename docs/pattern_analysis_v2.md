# Pattern Analysis V2

Pattern Analysis V2 is the default hardened pattern-analysis workflow.

```bash
python3 -m uwos.pattern_analysis_v2 \
  --base-dir /Users/anuppamvi/uw_root/tradedesk \
  --as-of latest
```

The default output path is:

```text
/Users/anuppamvi/uw_root/tradedesk/out/pattern_analysis_v2/YYYY-MM-DD
```

`python3 -m uwos.options_pattern_pipeline_v2` is a compatibility alias for the
same engine.

V2 emits ticket-first `AUTO_APPROVED`, `TRADE_REVIEW`, `AVOID`, and `NO_TRADE`
statuses plus decision-board, artifact-manifest, walk-forward, threshold,
calibration, shadow-ledger, profitability-audit, and runbook artifacts.

The implementation reuses the hardened source-first pattern engine while
branding artifacts as `pattern_analysis_v2.0`. The frozen V1 backup remains
unchanged.
