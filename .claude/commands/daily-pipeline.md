---
name: daily-pipeline
description: Run the daily trade playbook pipeline (risk monitor, open positions, daily edge report) and present results with clickable file links.
---

# /daily-pipeline Skill

Run the daily risk and trade playbook pipeline — pulls open positions, evaluates daily risk, and generates a structured daily report.

## Usage

| Command | Description |
|---|---|
| `/daily-pipeline` | Run daily pipeline for today |
| `/daily-pipeline 2026-03-08` | Run for a specific date |
| `/daily-pipeline 2026-03-08 schwab` | Run with live Schwab positions |

## Execution Steps

1. **Parse user arguments**
   - `date`: first arg (YYYY-MM-DD) — default: today
   - `source`: second arg — `schwab` or `local` (default: `local`)

2. **Determine paths**
   - `base-dir`: `c:/uw_root/{date}/`
   - `out-dir`: `c:/uw_root/out/playbook/`
   - `config`: `c:/uw_root/uwos/rulebook_config.yaml`

3. **Run the pipeline**

```bash
python -m uwos.report daily \
  --date {date} \
  --base-dir "c:/uw_root/{date}/" \
  --out-dir "c:/uw_root/out/playbook/" \
  --config "c:/uw_root/uwos/rulebook_config.yaml" \
  --open-positions-source {source}
```

   Add `--open-positions-source schwab` only if user explicitly requested Schwab live data.

4. **Read output files and present results**
   - Read the generated markdown report(s)
   - Present a structured summary of daily risk status, open positions, and any flagged setups
   - Always provide clickable file links to ALL output files

## Output Files

| File | Description |
|---|---|
| `c:\uw_root\out\playbook\daily_risk_report_{date}.md` | Daily risk monitor report |
| `c:\uw_root\out\playbook\daily_playbook_{date}.md` | Full daily trade playbook |

## Reference

- Pipeline code: `c:\uw_root\uwos\report.py`
- Entry shim: `c:\uw_root\uwos\build_day.py`
- Config: `c:\uw_root\uwos\rulebook_config.yaml`
