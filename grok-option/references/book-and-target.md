# Book and target

**Bar:** **$10k profit / month** ($120k/year). Stated. That is the scoreboard. It is not a size-up button and not a daily-fill quota.

**Account:** live Schwab when pulled. **$715k** on 2026-08-26, cash ~$24k. Mixed book: equity + options overlay. Size % off that total unless the user names an options-only sleeve. $150k is only if equity is unknown **and** Schwab is down.

**No protected-ticker list.** Do not skip a call credit or iron condor because a name is “core” or because Schwab shows long shares. Permission is **this session’s signals**: regime geometry, earnings-overlap, Event/Crowded, quotes, theme cap. If both credit wings pass the same name/expiry, print **one iron condor**. If Schwab shows long stock in that name, tag Notes `shares held — close the spread; do not deliver unless asked`. Early assignment is a manage item, not a SCAN veto. Naked short calls stay banned.

## What $10k/month is on this account

$10k / $715k = **1.4% of the whole account per month**.

| Engine | Floor $10k every calendar month? | Role |
|--------|----------------------------------|------|
| Whole account (equity + options) | No. Some months yes, some no. 90-day unrealized was +$102k, almost all equity | The bar is coherent as a **year average** (~$120k) |
| Options overlay, Calm VIX ~15 | No. 12% width at 1% risk cannot print $10k in 4–8 closes | Harvest + Score-65 Shields that clear |
| Options overlay, Elevated vol | Closer. 25% width is when credits can carry a month | Lean in when vol pays |

Do not add names, punch delta to 0.30, or raise per-name risk to manufacture a green month.
Do close 85%+ shorts, trade allowed Shield today, keep Fire/Spike small, journal toward $120k/year.

Calm 12% math (order of magnitude, not a promise): 1.0% of $715k ≈ $7.2k max loss; credit ≈ 14% of that ≈ $1.0k; take 65% ≈ $650. Four to eight closes do not equal $10k. That is the constraint.

New size is also capped by **buying power**. Cash was ~$24k with 25 option lines already open. Per-name 1.0% is a ceiling, not a fill. Do not 2× lots into a margin bind.

**Picker:** among gate-clearing credits, **credit/width first**, then dollars. 1-lot credit ≥ $100 when it exists. Do not take extra width that cuts edge. Rec lots scale from Conf only on those rows.

**Lot count on the card:** table dollars are always **1 lot**. Rec lots scale from Conf, then the sleeve cap, then cash/BP:

`Rec lots = max(1, round(naive_POP/100 × floor(sleeve_cap / 1-lot max loss)))`

Then `min` with remaining cash/BP. Rec lots is per row — do not assume every printed row is filled at that size.

This month’s options path starts with **open-book harvest**, not only new paper. Unrealized options were ~+$8.3k at the 90-day pull; several shorts are 85%+.

## $715k caps (live Schwab)

| Sleeve | Per name | Notes |
|--------|----------|--------|
| Shield | **1.0–1.5%** | Calm default **1.0%** ≈ $7.2k max loss, then BP. Legal **14–60 DTE**, prefer 21–45. Max **4** names, **one per sector** |
| Fire | 0.5–0.75% | Live on non-event names. Skip expiry on the index-event date; no SPY/QQQ Fire that day |
| Spike | 0.25% | Crude **or** sourced kinetic/defense. One name |
| Aggregate open risk | **6–8%** | $43k–$57k ceiling. Existing lines count. Cash ~$24k is the live bind |
| Theme | 30% of aggregate | Semis ≠ healthcare ≠ energy |
| Closes / month | 4–8 | Empty days count when geometry fails. Do not add tickets to chase $10k |

If 20-trade rolling expectancy < 0: **freeze Fire and Spike, cut size 50%**, AUDIT. Do not add trades.

## Default $150k reference caps (only if equity unknown and Schwab down)

| Sleeve | Per name | Notes |
|--------|----------|--------|
| Shield | 1.5–2.0% of equity | Max loss dollars, not notional |
| Fire | 0.75–1.0% | Half of Shield. One live Fire per name |
| Lotto / Spike | 0.25% | Sleeve D. One Spike name. No Prime if Crowded or THIN |
| Aggregate open risk | 10–12% | Sum of max losses |
| Theme | 35–40% of aggregate | AI/semi/cloud is one theme; XOM/CVX/COP is one energy theme |
| Closes / month | 6–12 | Skip days count as zero, not failure |

## Equity-band scaling

| Equity | Shield / name | Fire | Notes |
|--------|---------------|------|-------|
| < $50k | 1.0% max | Off, or 0.5% once | One live name. No lotto, **no Spike** |
| $50k–$150k | 1.0–1.5% | 0.5–0.75% | Still one Fire max |
| **$150k reference** | 1.5–2.0% | 0.75–1.0% | Only if unknown and Schwab down |
| **~$715k (live Schwab)** | 1.0–1.5% | 0.5–0.75% | $10k/month = 1.4% of account. Aggregate 6–8%. BP binds first |
| > $1M | keep % caps | keep % caps | $10k/month is ~1%/month |

## Correlation

Theme is **this session’s tape**, not a saved ticker list. Names the market is pricing as one cluster (AI/semi/cloud while a mega-cap chip/infra print is live; energy while a shock map is live) share **one** theme cap (30% of aggregate). Do not size those as independent Shield/Fire. Spike is still one name on a live energy or defense map (`spike.md`).

## Cash

Cash is Sleeve C. Empty table + cash is valid when **geometry** fails. It is not valid because a later-week event exists.

**Stuck vs correct empty**

- Correct: VIX Calm, no allowed name clears Calm geometry. Do not invent rows.
- Stuck: an allowed name is skipped because a semi print or Friday speech exists elsewhere.
- Harvest (extra, not a gate): after a name/theme print, re-scan **that cluster** next session. After an index event, re-scan **index Fire** next session.

Calm 12% / 22-delta is the Shield product in cheap vol. Score 65. Scanner prints it. Index put-credit / iron condor stays off until the user allows it.
