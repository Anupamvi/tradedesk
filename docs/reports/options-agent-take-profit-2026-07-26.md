# Options Agent — credit take-profit 0.80 → 0.70

**Date:** 2026-07-26
**Change:** `uwos/options_agent/core.py` → `CREDIT_TAKE_PROFIT_REMAINING = 0.70` (was `0.80`)
**Status:** validated end-to-end, uncommitted in the working tree

---

## Result

Buy the spread back when 30% of the credit remains instead of 20%. Nothing else changes — same
signals, same selection, same entries, same fills.

Full-window independent replay, 104 signal days (2026-01-02 → 2026-06-02), identical coverage to the
baseline (`detail_rows = 13245` in both runs), manifest confirms `credit_take_profit_remaining: 0.7`.

| | n | win% | PF | total $ | avg win | avg loss | hold |
|---|---|---|---|---|---|---|---|
| v1_57 — tp 0.80 | 783 | 89.8% | 1.798 | $13,755 | $44.07 | −$215.35 | 7.1 |
| v1_60 — tp 0.70 | 781 | 88.7% | **1.935** | **$17,706** | $52.89 | −$215.27 | 8.2 |

**+$3,951 (+28.7%).**

| segment | tp 0.80 | tp 0.70 | delta |
|---|---|---|---|
| train (< 2026-05-01) | $11,564 · PF 1.819 | $14,719 · PF 1.929 | +$3,155 |
| held-out (≥ 2026-05-01) | $2,191 · PF 1.705 | $2,987 · PF **1.962** | **+$796** |

Better in **6/6 months** — Jan +458, Feb +518, Mar +1,447, Apr +733, May +604, Jun +192.

Paired on identical trade *and* identical executed fill: n=781, **+$5.12/trade, 199 better, 8 worse,
574 unchanged.**

`selected_for_policy` subset (what the live book actually trades): n=37 in both runs,
$1,582 → **$2,066 (+30.6%)**, PF 5.019 → 6.248, win rate 97.3% unchanged, held-out $173 → $260.

## Why it works

The credit book is perfectly bimodal. Every `take_profit` exit wins; every `time_exit` exit loses.

| exit | n | win% | avg P&L | total | avg hold |
|---|---|---|---|---|---|
| `take_profit` | 703 | 100% | +$44.07 | +$30,983 | 5.7d |
| `time_exit` | 80 | 0% | −$215.35 | −$17,228 | 19.7d |

Trades that reach the target essentially always reach it. Exiting at 20% of credit was banking a
winner early and leaving the rest on the table. Holding for 30% costs roughly one extra session and
raises the average win from $44.07 to $52.89.

**Tail risk does not change.** Worst single trade −$420 either way. Ten worst trades sum to −$3,883 vs
−$3,886. Average loss is identical. The extra P&L is entirely larger wins, not more risk taken.

## Robustness

- Week-block bootstrap of the paired delta: 5th percentile **+$1,799**, P(improvement) **99.8%**.
- Not concentrated — the top five improving trades are 23% of the gain.
- 0.70 sits on a plateau: 0.65, 0.60 and 0.55 also clear the bootstrap. 0.75 and 0.50 do not.
  0.70 is the least aggressive point that wins on every axis, and the smallest deviation from the
  current setting.
- Offline sweep predicted $17,295; the real replay produced $17,706. The harness under-promises.

**Do not auto-tune this.** Selecting the level greedily on trailing months picks 0.30 and *loses*
13.8%. The fixed 0.70 is what is validated.

## The one metric that is nominally worse

Equity-curve drawdown, 1x contracts, ordered by exit day:

| | total | max DD | DD / total | longest underwater | worst week |
|---|---|---|---|---|---|
| tp 0.80 | $13,755 | −$5,196 | 0.38 | 146 trades | −$4,387 |
| tp 0.70 | $17,295 | −$5,617 | **0.32** | **132 trades** | −$4,938 |

Absolute drawdown is ~8% larger, but you earn 26% more, so drawdown per dollar earned improves and
recovery is faster.

---

## Also settled — two things not to do

**Hard stops are strongly harmful.** `CREDIT_HARD_STOP_ENABLED = False` is evidence, not an oversight.

| stop | total $ | PF |
|---|---|---|
| none | **$13,755** | 1.798 |
| 4.00x | $3,345 | 1.128 |
| 3.00x | $1,061 | 1.039 |
| 2.50x | −$364 | 0.987 |
| 2.00x | **−$1,963** | 0.927 |
| 1.50x | −$6,134 | 0.765 |
| 1.25x | −$7,627 | 0.677 |

Monotone, and negative in every single month. The mechanism: 221 trades touch a 2.0x stop, but **141
of them (64%) recover into winners** — the stop destroys $22,366 of realised P&L. A defined-risk
vertical already caps its loss, so a stop only converts recoverable noise into a realised loss.

Capping the holding period fails identically: 25 sessions → $12,465, 20 → $10,000, 15 → $9,388,
10 → $5,397, all worse than holding.

**The 52-week headroom gate has been removed.** It was the previous headline result. It does not
survive honest validation: the run that appeared to confirm it executed a stale copy of `core.py` in
the other clone and returned byte-identical numbers to the baseline, so it tested nothing.

---

## Capacity — the honest number

At tp 0.70 the whole evaluated credit book averages $22.10/trade across ~130 trades/month, so roughly
**$2,900/month at 1x**. Monthly totals range $556–$6,866, and the worst month (January) is $3.59/trade.

$10k/month needs about 3.5–4x contracts, and even then January under-delivers. This remains a capital
and throughput problem, not an edge problem.

**Caveat on all held-out claims:** the out-of-sample window is structurally capped at about one month
(`_eligible_replay_days` requires `day + 35 sessions <= end`), so 2026-05-01 → 06-02 is only 22
trading days. The direction is consistent everywhere, but the out-of-sample sample is small.

---

## Before this affects anything live

1. **`codexuw/*.py` does not import `options_agent`.** The codexuw daily_v4 pipeline is a separate
   product with its own credit OCO policy. This change does not alter codexuw tickets.
2. **The two trees are separate clones on different branches.** The workspace is `main`;
   `~/uw_root/tradedesk` is `pipeline-profitability-fix`, and its `core.py` is far older — it has
   `PLANNED_TRADE_HOLDING_SESSIONS = 5` and no take-profit constant at all. Whichever tree
   `python -m uwos.options_agent` is launched from decides what is live.
3. **Nothing is committed.** `git diff` shows hold-to-expiry (5 → 35), the take-profit constant and
   the no-stop rule all as new working-tree lines. The entire credit management policy is uncommitted.

## Reproduce

```sh
cd /Users/anuppamvi/tradedesk
python3 -m uwos.options_agent.replay \
  --root /Users/anuppamvi/uw_root/tradedesk \
  --start 2026-01-02 --end 2026-07-23 --split-day 2026-05-01 \
  --output-dir /Users/anuppamvi/uw_root/tradedesk/out/options_agent_independent_replay/<name>
```

`--end` is the **observation** end, not the signal end; passing `2026-06-02` silently truncates the run
to 69 days with zero held-out rows. Always check `days` and `evaluated_rows` in the manifest against the
baseline before interpreting any delta.

Fast policy sweeps without a full replay (~2 min, reuses the recorded leg quotes):

```sh
cd /Users/anuppamvi/tradedesk
PYTHONPATH=/Users/anuppamvi/tradedesk python3 scripts/stop_loss_sweep.py \
  --replay-dir <abs replay dir> --root /Users/anuppamvi/uw_root/tradedesk --tp-sweep
```

The sweep reproduces the recorded baseline exactly (n=783, PF 1.798, $13,755) before reporting
anything — that check is the reason its predictions can be trusted.
