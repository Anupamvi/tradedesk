# Compound Core

CODE is this repository: `/Users/anuppamvi/tradedesk/compoundcore`.

Compound Core is the long-term **index sleeve**. Independent of groat, wheelo, xhigh, groko, and Codex Daily. No Python import of those desks.

When the user says `compound core`, `compoundcore`, `core sleeve`, `core calculator`, `core dashboard`, or pastes a dollar amount into this sleeve, read `skills/compoundcore/SKILL.md` and run it. Do not tell the user to type `python3` or start servers — run the calculator or **ensure the dashboard is running** and give them the link.

**Dashboard auto-start:** run `scripts/ensure-dashboard.sh` from CODE (or `scripts/install-macos-dashboard-service.sh` once on macOS so it survives reboot). On Cloud Agents the repo `.cursor/environment.json` starts it on port **8765**; use Cursor **Ports** on the agent if `127.0.0.1` is not on their machine.

No orders. No options. No leverage. Empty of trading-desk tickets is the point.

Never invent CMA numbers, holdings, or X posts. Missing source → **DATA UNAVAILABLE**.
