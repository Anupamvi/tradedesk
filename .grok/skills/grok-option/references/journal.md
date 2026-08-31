# Journal

The journal is how rules change. AUDIT is not a vibe pass.

## What to record on every close

`date | ticker | sleeve | action | expiry | net | max_loss | result_$ | result_R | days_held | skip_or_fill | notes`

Include **skips** that were Event/empty-table days (result empty, note `skip`). Otherwise the 20-trade window lies.

## Rolling 20-trade review (weekly, or at 20 closes)

Compute:

- win rate by sleeve (this is **book Conf**; do not advertise 70%)
- average win / average loss
- expectancy per $1 risk
- Shield vs Fire vs Spike split
- theme concentration at time of entry
- how many rows were THIN

If **20-trade expectancy < 0**: freeze Fire and Spike, cut size 50%, open AUDIT, rewrite `assumption-audit.md` for any contradicted rule.

If **5+ consecutive empty sessions** while non-event names exist and VIX is still Calm: that is a stuck gate or a dead Calm regime — rescan 14–60 DTE, do **not** drop below 0.22-delta / 0.80-sigma. If still empty, cash is the trade.

If expectancy > 0 but theme cap was breached: keep size, tighten clustering — do not celebrate.

## Weekly assumption review

Read the numbered verdicts. Change a rule only with journal evidence or a sourced tape change (new VIX regime, new paper, persistent quote failure). Date the edit. Do not silently restore Anu v1 theater (74.0, estimated IV, always-a-table, "Buy Put Credit", X-as-entry).

## Attachments that leave THIN

To reach MIXED/FULL next scan, attach some of:

- same-day option chain CSV or broker screenshot with bid/ask both legs
- account equity and open max-loss by name
- earnings calendar export (confirmed dates; BMO/AMC if listed)
- flow file with open/close if you have it (optional; still not a trigger)
- headed Playwright login to x.com in `~/.grok/browser/grok-option` (optional; still not a trigger)

Web/X alone stays THIN. THIN may still print Shield on mega-caps **if** both legs are quote-verified from the web; it may not Prime.
