---
name: wheel
description: Run the Wheel strategy pipeline (selection, daily management, or full) and present results with clickable file links.
---

# /wheel Skill

Run the wheel options pipeline for candidate selection, daily position management, or both.

## Usage

| Command | Mode | Description |
|---|---|---|
| `/wheel` | full | Run both selection and daily management (capital=35000) |
| `/wheel select` | select | Run candidate selection only |
| `/wheel daily` | daily | Run daily position management only |
| `/wheel select 50000` | select | Run selection with custom capital ($50,000) |

## Execution Steps

1. **Parse user arguments**
   - `mode`: first word arg — one of `select`, `daily`, `full`. Default: `full`
   - `capital`: numeric arg — default `35000`
   - `as-of`: date arg (YYYY-MM-DD) — default: today

2. **Determine paths**
   - `base-dir`: `c:/uw_root/{as-of}/`
   - `out-dir`: `c:/uw_root/out/wheel/`

3. **Run the pipeline**

```bash
python -m uwos.wheel_pipeline \
  --mode {mode} \
  --capital {capital} \
  --base-dir "c:/uw_root/{as-of}/" \
  --out-dir "c:/uw_root/out/wheel/" \
  --as-of {as-of}
```

   Add `--no-schwab` only if the user explicitly requests it.

4. **Read output files and present results**
   - Read the generated markdown report(s)
   - Present a summary table of top candidates or position actions
   - Always provide clickable file links to ALL output files

## Output Files

| File | Description |
|---|---|
| `c:\uw_root\out\wheel\wheel-select-{date}.md` | Selection report — ranked wheel candidates |
| `c:\uw_root\out\wheel\wheel-daily-{date}.md` | Daily management report — position actions |
| `c:\uw_root\out\wheel\wheel_positions.json` | Persistent position state |

## Reference

- Design doc: `c:\uw_root\docs\plans\2026-03-07-wheel-pipeline-design.md`
- Pipeline code: `c:\uw_root\uwos\wheel_pipeline.py`
- Config: `c:\uw_root\uwos\wheel_config.yaml`
