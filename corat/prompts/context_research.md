# CORAT dated context-research prompt

Prepare a `corat.context.v1` JSON evidence file for the requested decision date and ticker set.

Use the smallest sufficient source set:

1. Company filings, investor-relations releases, official company accounts, regulators, and official economic calendars.
2. Reputable financial or industry reporting for developments that do not yet have a primary source.
3. X only for discovery, narrative acceleration, customer/supplier color, or clearly labeled rumor.

For every row preserve the direct source URL and publication date. Separate `FACT`, `REPORTED INFORMATION`, and `RUMOR / X SPECULATION`. Never copy a social claim into `FACT` unless the original primary source confirms it.

For X, search relevant 1-hour, 4-hour, 24-hour, 3-day, and 7-day windows when the tool permits. Downweight anonymous promotional accounts, repeated wording, low-history accounts, referral links, engagement farming, and unsupported ticker promotion. Mark `spam_risk: true` when warranted. Mention acceleration is a discovery observation, not a catalyst.

For each candidate ask:

- What changed recently?
- Is it directionally relevant to this ticker?
- Is it new, developing, potentially under-appreciated, fully priced, or stale?
- What is the original source?
- What would prove or disprove the proposed interpretation?
- Which macro, earnings, product, regulatory, legal, conference, rebalance, or other dated event falls inside a typical 10-session hold?

Use `python3 -m corat context-template --date DATE --tickers TICKERS` to create the empty structure. Return valid JSON only in the saved file. Do not include unavailable claims, fabricate URLs, invent X posts, or infer dealer positioning from open interest.
