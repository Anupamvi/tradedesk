# Research basis and source decisions

## Can this pipeline be profitable?

Possibly, but neither more data nor AI creates edge by itself. A defensible edge
must survive realistic fills, costs, overlapping outcomes, regime changes,
multiple testing, and an untouched holdout. A current trade must then remain
positive when mapped to an exact liquid contract and the actual portfolio.

The pipeline is therefore built to falsify candidates cheaply and promote only
the survivors. It is not built to guarantee that every day has a trade.

## Why ORATS is the primary research source

ORATS's historical EOD API exposes adjusted daily stock prices and exact option
strikes, quotes, deltas, volume, and open interest back to 2007. Historical
strikes require a ticker and trade date, support up to ten comma-delimited
tickers, and allow DTE/delta filters. That makes exact-structure replay possible
without minute data. See the [ORATS Historical Data API](https://orats.com/docs/historical-data-api).

The delayed `cores` feed supports broad discovery and contains both
`orFcst20d` and `orIvFcst20d`; the former is the forecast underlying realized
volatility and the latter is forecast option implied volatility. See the
[ORATS Delayed Data API](https://orats.com/docs/delayed-data-api).

ORATS describes conservative multi-leg backtests and warns about inaccurate
fills, overfitting, path dependence, and comparing notional with margin. The
implemented two-leg entry uses 66% of package width from the favorable side;
the exit is stricter at natural liquidation. See [ORATS Backtesting Methodology](https://orats.com/university/backtesting-methodology)
and [Custom Backtesting](https://orats.com/university/custom-backtesting).

ORATS's Trade Ideas product can search a large set of precompiled backtests,
which validates the workflow concept: current environment -> matching strategy
history -> candidate. CodexSwing does not ingest that vendor feed. It performs
its own fixed-rule discovery, exact replay, and Schwab verification so the
evidence is inspectable. See [ORATS Trade Ideas and Signals](https://orats.com/university/trade-ideas-and-signals).

## Why Schwab is current execution truth

Delayed or theoretical option data is suitable for discovery and historical
research, not for deciding an exact current order. Schwab provides the current
regular-session stock quote, exact option symbols and chain, and the user's
current balances, positions, and working orders. CodexSwing reads those
surfaces with GET only. It deliberately has no submit, replace, or cancel
method.

## Internet trends and geopolitical information

Search/news attention can be related to trading volume, volatility, and market
moves. Examples include the open research on [Google Trends and trading behavior](https://www.nature.com/articles/srep01684),
[Wikipedia usage before market moves](https://www.nature.com/articles/srep01801),
and [financial-news mentions and market activity](https://www.nature.com/articles/srep03578).

Those findings do not justify adding an untested sentiment score to today's
rank. Attention data is revised, keyword-sensitive, correlated across names,
and vulnerable to publication and selection bias. CodexSwing stores current
source-cited public context but gives it no numeric vote. Promotion requires an
as-of timestamp, fixed extraction schema, explicit missing-data policy, and
incremental holdout improvement versus the ORATS/Schwab baseline.

## Backtest design

- Signal features: adjusted ORATS stock path plus as-of IV regime.
- Current conditional cohort: 250 nearest prior same-direction environments,
  selected without outcome labels.
- Entry: next-session trigger, reject >1% gaps and same-day invalidations, price
  the option at next-session EOD.
- Strategies: long call, long put, bull call debit, bear put debit, bull put
  credit, and bear call credit.
- Contract: 21–60 DTE; fixed delta, width, liquidity, and quote-width ranges.
- Fill: single legs enter 75% from bid toward ask and exit at exact bid;
  verticals enter 66% into the package and exit at exact natural liquidation;
  $0.65 per contract per leg per side.
- Split: chronological 50% train, 20% validation, 30% untouched holdout.
- Dependence: report both closed trades and effective non-overlapping trades;
  bootstrap by five-session clusters.
- Holdout gate: at least 20 closed and 8 effective trades, positive mean and
  positive 2.5% bootstrap lower mean, profit factor >=1.10, Wilson POP lower
  bound >=40%, and positive validation/temporal stability.
- Current gate: positive modeled EV at the explicit limit, <=25% maximum leg
  relative spread, OI >=100, volume >=10, defined max loss, and fresh Schwab
  session evidence.
- Portfolio gate: sufficient available funds, candidate max loss <= min($2,000,
  1% liquidation value), concentration <10%, and no same-underlying option or
  working-order conflict.
- Tactical gate: at least 30 holdout closes / 15 independent outcomes, positive
  train-validation-holdout expectancy, validation/holdout PF >=1.20, uncertainty
  no worse than 5% of current defined risk, and one contract <=min($500, 0.05%
  NAV). Tactical evidence is exploratory and is not full-size validation.

## Selection-bias control

Testing many variants and reporting only the winner inflates performance. The
[Deflated Sharpe Ratio paper](https://doi.org/10.2139/ssrn.2460551) explicitly
addresses selection bias, non-normal returns, and backtest overfitting. v0.3
limits itself to four predeclared structures and does not optimize against the
holdout. Any expansion must record the number of attempted variants and add a
multiple-testing correction.

## Interpretation of a no-trade result

No trade is not evidence that the APIs are useless. It can mean:

- current model economics look attractive but exact historical economics do
  not survive;
- point POP is positive but its uncertainty is too large;
- a structure works in validation but not in train/holdout;
- current quotes are too wide or thin;
- the account already has conflicting exposure; or
- no tested fixed structure has edge in the current regime.

The response is to test additional predeclared hypotheses or wait for a better
setup—not to lower gates until something passes.
