# Codex Daily V4.21 Frozen Baseline

Version: `v4.21-directional-credit-medium-book-20260812`
Frozen: 2026-08-13

Validated production authority: directional credit Medium book only.

Evidence:
- 83 corrected next-session trades
- 90.36% win rate
- 82.12% Wilson 90% lower bound
- PF 3.579
- P/L +$5,096.50 per one-contract sequence
- max drawdown -$603.50
- 7 of 8 positive months

Disabled:
- Bull Call debit, corrected PF 0.726
- Bear Put debit, corrected PF 0.828
- legacy strict credit selector after impossible-credit cleanup
- generic all-strategy walk-forward selector

Rollback: extract `codexdaily-v4.21-source-tests.tar.gz` from the tradedesk root, restore `codexdaily-SKILL.md` to `~/.codex/skills/codexdaily/SKILL.md`, then run the focused V4 tests.
