# How AI should be used in CodexSwing

## What practitioners and researchers are doing

Financial language models are used for document classification, entity/event
extraction, sentiment, question answering, summarization, and analyst workflow
automation. [BloombergGPT](https://arxiv.org/abs/2303.17564) demonstrates a
domain model trained on mixed financial/general corpora; [FinGPT](https://arxiv.org/abs/2306.06031)
emphasizes transparent data curation and lightweight adaptation.

Recent research also experiments with multi-agent investment teams. The
[TradingAgents paper](https://arxiv.org/abs/2412.20138) separates fundamental,
sentiment, technical, bull/bear, trader, and risk roles. A newer
[fine-grained expert-team study](https://arxiv.org/abs/2602.23330) argues that
specific task decomposition and alignment between intermediate analysis and
the downstream decision rule matter more than vague analyst personas.

These are research results, not proof that an LLM can generate durable retail
trading profits. Backtests can leak future information, select favorable
prompts, ignore execution, or count correlated observations as independent.

## The role AI gets here

AI is an evidence compiler and hypothesis generator, not the source of POP.

Useful tasks:

- map news/filings to ticker, event type, event time, source, novelty, expected
  direction, and confidence;
- produce bull and bear catalyst summaries with direct source links;
- detect conflicts between a current thesis, scheduled earnings/dividends, and
  geopolitical exposure;
- explain why a candidate failed a quantitative gate;
- propose a small number of predeclared features or strategy hypotheses for a
  future ablation;
- audit data timestamps, missingness, report consistency, and source roles.

Prohibited shortcuts:

- let an LLM assign an unsupported "70% confidence";
- feed current/future news into a historical row;
- optimize prompts or thresholds on untouched holdout;
- convert prose sentiment directly into position size;
- allow an agent debate to overrule negative exact-chain expectancy;
- submit an order.

## Proposed event schema

Every AI-extracted context item should contain:

```text
event_id
ticker_or_macro_scope
event_time_utc
first_seen_time_utc
source_url
source_type
event_type
direction_hypothesis
novelty_score
extraction_confidence
evidence_excerpt_hash
model_and_prompt_version
```

The feature becomes eligible only if it could have existed before the decision
timestamp. Missing news is not neutral by default; missingness is a separate
feature and failure state.

## Promotion test for an AI feature

1. Freeze the extraction schema, model, prompt, universe, horizon, and cost
   model.
2. Backfill only from sources with defensible publication/first-seen times.
3. Compare the ORATS/Schwab baseline with baseline + one AI feature family.
4. Use chronological purged validation and untouched holdout.
5. Require positive incremental mean after costs, positive clustered-bootstrap
   lower incremental mean, stable sign across regimes, and no unacceptable
   turnover/concentration increase.
6. Record every attempted feature family for multiple-testing correction.
7. Keep the feature shadow-only if any requirement fails.

## Governance

FINRA notes that firms are exploring AI in portfolio management and trading and
emphasizes supervision, controls, validation, and model-risk governance. See
[FINRA AI applications](https://www.finra.org/rules-guidance/key-topics/fintech/report/artificial-intelligence-in-the-securities-industry/ai-apps-in-the-industry),
[FINRA Regulatory Notice 24-09](https://www.finra.org/rules-guidance/notices/24-09),
and [FINRA AI challenges](https://www.finra.org/rules-guidance/key-topics/fintech/report/artificial-intelligence-in-the-securities-industry/key-challenges).

The SEC has also warned against exaggerated claims that AI will produce better
returns; see its [AI-washing statement](https://www.sec.gov/newsroom/speeches-statements/sec-chair-gary-gensler-ai-washing).
CodexSwing therefore reports exactly what AI seeded and never markets AI itself
as edge.
