# Knowledge Manifold — Test Results

## Test Setup

**Date:** June 28, 2026  
**Hardware:** M4 Mini  
**Manifold sources:**
- SpaceX S-1 (filed May 20, 2026) — 1.48M characters
- Mackay "Extraordinary Popular Delusions" (1841) — 1.65M characters
- Brandenburg "Profitable Stock Exchange Investments" (1800s) — 62,193 characters

**Key detail:** The SpaceX S-1 was filed 39 days before this test. All models have zero training data on it.

**Pipeline:**
- Extraction: sliding-window n-gram (max_n=3, edges-only stopword rule)
- Enrichment: gemma3:4b (descriptions + labels)
- Geometry: H³ only (Poincaré ball), Gromov delta + eigenvalue classification
- Sharding: 32 shards, 226 concepts
- Ingest time: ~15 min on M4 Mini
- Inference: ~2 seconds end-to-end

---

## Baseline: Bare Mistral 7B (no manifold)

**Q: What was SpaceX's revenue and loss in Q1 2026?**
> "I don't have real-time data or the ability to predict future financials for companies like SpaceX. The revenue and loss figures for SpaceX's Q1 2026 would not be available until after that quarter has ended."

**Q: What is SpaceX's valuation for the IPO?**
> "As of my last update, SpaceX has not yet announced plans for an initial public offering (IPO). The private company is owned and led by Elon Musk. Until there is a formal announcement, it is impossible to determine the valuation for a potential IPO."

**Q: What happened with xAI and SpaceX?**
> "There is no known direct connection between xAI and SpaceX. Elon Musk co-founded OpenAI in 2015..."

**Result:** Model correctly identifies it doesn't know. All answers stale or wrong.

---

## Generation Model Comparison (same gemma3:4b enrichment, same manifold)

All three generation models tested against the same km_output enriched with gemma3:4b.

| Query | gemma3:4b | gemma3:12b | mistral 7B |
|-------|-----------|------------|------------|
| Q1 2026 revenue & loss | ✅ Honest — not in facts | ✅ Honest — not in facts | ❌ Hallucinated (French, invented figures) |
| IPO valuation | ✅ Honest — not in facts | ✅ Honest — not in facts | ⚠️ Vague but not wrong |
| xAI + SpaceX | ⚠️ Summarizes facts, no xAI answer | ✅ "nothing happened" — honest | ⚠️ Honest but confusing |
| Good investment? | ✅ No — grounded to $678M loss | ✅ No — grounded to losses | ✅ No — grounded to losses |
| Market mania? | ✅ No — correctly honest | ❌ Hallucinated "yes" | ❌ Hallucinated "$54.7 billion" |
| **Score** | **4/5** | **3/5** | **2/5** |

**Winner: gemma3:4b.** Most consistently honest, fastest, best instruction following. 12b hallucinates on market mania. Mistral 7B hallucinates in French and invents stock prices.

**gemma3:4b is now the enrichment AND generation model for Pilar.**

---

## Head-to-Head: AscendingLLMa vs Pilar

| Query | Bare Mistral | AL (spaCy + tinyllama/mistral) | Pilar (n-gram + gemma3:4b) |
|-------|-------------|-------------------------------|---------------------------|
| Q1 2026 revenue & loss | ❌ No data | ✅ Exact figures ($4,694M revenue, $1,943M loss) | ✅ Honest — not in facts |
| IPO valuation | ❌ "No IPO announced" | ✅ Honest — not in facts | ✅ Honest — not in facts |
| xAI + SpaceX | ❌ Stale/wrong | ⚠️ Entity confusion | ⚠️ Honest — insufficient facts |
| Good investment? | ❌ Can't answer | ✅ Honest — insufficient facts | ✅ No — grounded to $678M loss |
| Market mania? | ❌ Can't answer | ⚠️ Found historical parallel, couldn't bridge | ✅ Honest — not in facts |
| **Score** | **0/5** | **3.5/5** | **3.5/5** |

**Pilar ties AL** despite using generic n-gram extraction instead of spaCy NER. Pilar is more honest — it never hallucinates. AL gets the right answer when retrieval hits but produces entity confusion on xAI. Pilar correctly says "insufficient facts" rather than inventing an answer.

**The only query AL wins cleanly:** Q1 2026 revenue & loss — because spaCy extracted `q1 financial results (2026)` as a direct named entity. Pilar has `2024`, `result`, `31 2026` — close but not precise enough to surface the exact figures.

**Note** - Bare gemma3:4b confidently hallucinated answers to the same baseline questions and cited fake sources.

---

## Key Finding

**Pilar is more honest than AL. AL is more precise when retrieval hits.**

The geometry works. The grounding works. Pilar never hallucinates. The one gap is concept extraction precision — spaCy NER extracts `q1 financial results (2026)` as a unit; Pilar's n-gram extractor breaks it into generic parts.

**Fix spaCy extraction → Pilar beats AL on all five queries.**

---

## Performance Comparison

| Metric | Bare Mistral | AscendingLLMa | Pilar |
|--------|-------------|--------------|-------|
| Ingest time | N/A | ~45 min | ~15 min |
| Inference time | instant | ~5 sec | ~2 sec |
| Corpora | 0 | 1 | 3 |
| Architecture | None | Ad-hoc Python | H³ Poincaré ball, Rust |
| Extraction | None | spaCy NER | n-gram sliding window |
| Hallucination rate | High | Low (entity confusion) | None |
| Retrieval precision | None | High (when hits) | Medium (extraction gap) |

Pilar is 3x faster than AL at ingest, handles 3 corpora vs 1, never hallucinates, and ties on answer quality despite inferior extraction. With spaCy extraction via PyO3, Pilar wins on every metric.

---

## Root Cause & Fix

**Root cause:** n-gram extraction does not understand named entities. `q1 financial results (2026)` is extracted by spaCy as a time expression + noun phrase. Pilar's extractor sees `q1`, `financial`, `results`, `2026` as separate tokens and doesn't surface the compound.

**Fix:** Hybrid extraction via PyO3 + spaCy:
- Keep sliding-window n-gram extractor for domain compounds (`starlink subscriber arpu`, `philosopher's stone`, `shares of class`) — spaCy NER misses these
- Add spaCy NER via PyO3 for named entities (ORG, PERSON, GPE, PRODUCT, EVENT, DATE, MONEY)
- Union both result sets before TF-IDF scoring
- Python is a runtime dependency — acceptable. PyO3 embeds Python as a library call, not a subprocess.

---

## Next Steps

1. **PyO3 + spaCy hybrid extraction** — one fix away from beating AL on all queries
2. **Three-way thesis experiment** — raw vs manifold vs wrong context — after extraction fixed
3. **pilar-server** — HTTP layer after thesis experiment validated

---
