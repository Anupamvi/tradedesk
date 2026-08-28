# CORAT evidence-context contract

`corat.context.v1` is the boundary between sourced research and deterministic scoring. CORAT's automatic news adapter, a browser-capable research agent, or an analyst can supply dated catalysts, event risk, X observations, and options-flow evidence without letting the pipeline invent facts.

The machine-readable contract is [context.schema.json](../schemas/context.schema.json).

## Required top-level fields

- `schema_version`: exactly `corat.context.v1`.
- `as_of`: the research cutoff in `YYYY-MM-DD` form. It may not be after the CORAT decision date.
- `market_events`: sourced macro or market-wide events.
- `tickers`: an object keyed by uppercase ticker.
- `research_metadata` is optional and records automatic research coverage/errors; it never counts as evidence itself.

Each ticker can contain:

- `catalysts`
- `events`
- `x_intelligence`
- `options_flow`
- `mention_acceleration`

## Every evidence row requires

- `classification`: `FACT`, `REPORTED INFORMATION`, or `RUMOR / X SPECULATION`.
- `credibility`: `PRIMARY`, `HIGH`, `MEDIUM`, or `LOW`.
- `source`: human-readable publisher/account/document name.
- `source_url`: direct HTTP(S) URL.
- `published_at`: source-publication date.
- `title` or `claim`: concise description of what the source supports.
- `direction`: `BULLISH`, `BEARISH`, `NEUTRAL`, `MIXED`, or `UNKNOWN`.
- `event_date` when the row describes a dated future event.

Optional `freshness` values are `NEW`, `DEVELOPING`, `KNOWN BUT UNDER-APPRECIATED`, `FULLY PRICED`, or `STALE`. If omitted, CORAT derives freshness from `published_at` and the decision date. It normalizes `KNOWN BUT UNDER-APPRECIATED` to `KNOWN BUT POTENTIALLY UNDER-APPRECIATED` internally.

## Evidence-use rules

- Only `FACT` or `REPORTED INFORMATION` from `PRIMARY` or `HIGH` credibility sources receive full catalyst weight.
- The catalyst must be `NEW`, `DEVELOPING`, or `KNOWN BUT ... UNDER-APPRECIATED`.
- Catalyst direction must match the proposed trade direction.
- Rumor/X-only evidence can assist discovery or ranking context but cannot masquerade as a verified catalyst or create expected profit.
- A supplied future `published_at` date is rejected. A future `event_date` is allowed and disclosed when it falls inside the planned holding window.
- Absence of an event row means `DATA UNAVAILABLE`, not proof that no event exists.
- Missing catalyst evidence is disclosed and reduces ranking context; it does not mechanically veto a triggered plan whose measured POP/expected profit is positive.

## Minimal example

```json
{
  "schema_version": "corat.context.v1",
  "as_of": "2026-08-27",
  "market_events": [],
  "tickers": {
    "EXAMPLE": {
      "catalysts": [
        {
          "classification": "FACT",
          "credibility": "PRIMARY",
          "source": "Example issuer IR",
          "source_url": "https://example.com/ir/release",
          "published_at": "2026-08-27",
          "title": "Example sourced development",
          "direction": "BULLISH"
        }
      ],
      "events": [],
      "x_intelligence": [],
      "options_flow": [],
      "mention_acceleration": "DATA UNAVAILABLE"
    }
  }
}
```

The example is structural only; it is not market evidence and must never be used in a scan.
