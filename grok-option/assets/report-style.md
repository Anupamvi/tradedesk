# Report visual language

Plain GitHub-flavored markdown only. **No HTML.** No `<span>`, `<div>`, `<h1 style>`, or `&nbsp;` in cards. Those tags print as junk in the editor.

Use markdown headings for size (`#` title, `##` sections). GitHub alerts (`> [!NOTE]` / `WARNING` / `IMPORTANT` / `TIP`) are allowed. Tables: short cells, **icon in the first column**, no tags inside pipes.

## Color / icon key (every card)

Put this legend once under the title.

🟢 send-list / Expert · 🟡 review (geometry-pass, parked) · 🔴 blocked / failed · 🔵 Fire quote · ⚡ Spike

| Mark | Meaning | Write this |
|------|---------|------------|
| 🟢 | Expert row — re-quote then send | `🟢` in Status / first col |
| 🟡 | Cleared, parked — reviewable | `🟡` + short reason |
| 🔴 | Failed gate / do not add | `🔴` |
| 🔵 | Fire quoted, no flow | `🔵` |
| ⚡ | Spike lane | `⚡` |
| 🛡️ | Shield | `🛡️ Shield` |
| 😌 | Calm | `😌 **Calm**` |
| 😐 | Normal | `😐 **Normal**` |
| ⚠️ | Elevated or worse-fill | `⚠️` |
| 🚨 | Crisis | `🚨 **Crisis**` |
| 📗 FULL | both legs + sourced earnings | `📗 FULL` |
| 📒 MIXED | quotes yes, something estimated | `📒 MIXED` |
| 📕 THIN | skip Prime | `📕 THIN` |
| 🤫 / 📢 / 📰 | X Quiet / Crowded / Event | `X: 🤫 Quiet` |
| 📦 | shares held | `📦 shares held` |
| ⛔ | do not add (live overlay) | `⛔` |
| 📈 / 📉 | tape up / down | `📈 +2.87%` / `📉 −1.12%` |
| CLOSE / HOLD / ASSESS / ROLL | manage | **CLOSE** etc. |
| P&L up / down | dollars | `+$1,234` / `−$1,234` |

Do not invent extra colors. Do not use HTML color spans.

## Layout

1. Title + one-line **legend**
2. Dashboard (equity · regime · Expert/Review/Fail counts)
3. Alerts (`> [!IMPORTANT]` 9:35 re-quote, `> [!WARNING]` fragile wing)
4. **🟢 Expert** — one compact **card per ticket** (legs, credit, 1-lot P/L, Rec, Conf), then a slim compare table
5. Shock watch
6. Sleeve board (icons, not a one-line dump)
7. **🟡 Review** full rows
8. **🔵 Fire**
9. **🔴 Failed**
10. Tape · Book · Assumptions · Watch

Expert cards beat a 12-column wall for the send-list. Review still uses a table so every parked trade is inspectable.

Print order stays: Expert, Shock, sleeve, review. The dated file `out/grok-option/YYYY-MM-DD/GROK_OPTION.md` is the report. Chat without that file is incomplete. Reply with a clickable path.
