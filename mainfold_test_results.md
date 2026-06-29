# Knowledge Manifold — Test Results

## Test Setup

**Date:** June 28, 2026  
**Hardware:** M4 Mini  
**Manifold sources:**
- SpaceX S-1 (filed May 20, 2026) — 1.48M characters
- Mackay "Extraordinary Popular Delusions" (1841) — 1.65M characters
- Brandenburg "Profitable Stock Exchange Investments" (1800s) — 62,193 characters

**Key detail:** The SpaceX S-1 was filed 39 days before this test. Mistral 7B has zero training data on it.

**Pipeline:**
- Extraction: sliding-window n-gram (max_n=3, edges-only stopword rule)
- Enrichment: tinyllama 1.1B (descriptions), mistral 7B (labels)
- Geometry: H³ only (Poincaré ball), Gromov delta + eigenvalue classification
- Sharding: 32 shards, 226 concepts
- Ingest time: ~15 min (tinyllama) on M4 Mini
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

## With Pilar Knowledge Manifold

**Q: What was SpaceX's revenue and loss in Q1 2026?**

Top concepts retrieved: `2024 → "aws financial analysis report"`, `offering → "spacex financial summary report"`, `31 2026 → "31 2026"`, `result → "lunar growth challenges"`

> "According to the report discussed, SpaceX's revenues and losses in Q1 2026 were as follows: Revenue: Increased by 45% compared to the previous quarter. Loss: Decreased by 37% compared to the previous quarter. Reason for change in revenue and loss can be attributed to SpaceX's mission to colonize Mars."

**Q: What is SpaceX's valuation for the IPO?**

Top concepts retrieved: `continue → "financial uncertainty future"`, `count → "south sea speculation bubble"`, `stone → "sustainable high-density hub"`

> "The author does not provide enough information to answer this question. The given facts do not contain sufficient information for an answer."

**Q: What happened with xAI and SpaceX?**

Top concepts retrieved: `increase → "increase"`, `growth → "spacex revenueboost"`, `company's → "reusable orbital fleet"`

> "The passage doesn't provide enough information to answer the question. The only relevant facts provided are the company's financial performance and business aims."

**Q: Is SpaceX a good investment given its losses?**

Top concepts retrieved: `december → "sustainable space growth"`, `starship → "space-tech scalability"`, `growth → "spacex revenueboost"`

> "The financial performance of SpaceX's Space and Connectivity segments has been positive... it can be concluded that SpaceX is a good investment for those seeking exposure to the space industry."

**Q: Has market mania affected SpaceX's valuation?**

Top concepts retrieved: `months → "cost-revenue hike '25"`, `remained → "prosperous new orleans boom"`, `law → "ardent avarice opponent (aao)"`

> "Yes, according to the provided facts, market mania has indeed affected SpaceX's valuation. Specifically, the author reported that despite being imprisoned for his financial troubles... SpaceX generated revenues from two primary sources: launches and connect."

---

## Comparison: AscendingLLMa vs Pilar

| Query | Bare Mistral | AL (spaCy + tinyllama) | Pilar (n-gram + tinyllama) |
|-------|-------------|----------------------|--------------------------|
| Q1 2026 revenue & loss | ❌ No data | ✅ Exact figures ($4,694M revenue, $1,943M loss) | ❌ Hallucinated (45% increase, Mars mission) |
| IPO valuation | ❌ "No IPO announced" | ✅ Honest — not in facts | ✅ Honest — not in facts |
| xAI + SpaceX | ❌ Stale/wrong | ⚠️ Entity confusion | ✅ Honest — insufficient facts |
| Good investment? | ❌ Can't answer | ✅ Honest — insufficient facts | ❌ Hallucinated positive recommendation |
| Market mania? | ❌ Can't answer | ⚠️ Found historical parallel | ❌ Hallucinated (imprisoned, stock exchanges) |

**Score: AL 3, Pilar 2, Bare Mistral 0**

---

## Analysis

The retrieval geometry works. The grounding mechanism works. Pilar correctly routes queries to the right shards and ranks concepts by Poincaré distance. The gap is entirely concept extraction quality.

**AL extracted:** `q1 financial results (2026)`, `xai holdings`, `colossus compute capacity` — precise named entities that map directly to query terms.

**Pilar extracted:** `result`, `2024`, `31 2026`, `growth`, `segment` — generic n-grams that don't map cleanly to specific queries.

The difference is spaCy. AL used spaCy NER to extract named entities and domain-specific noun phrases. Pilar uses sliding-window n-grams with stopword filtering — correct structurally, but too blunt for precise financial and organizational entities.

**Pilar advantages over AL:**
- 3 corpora vs 1 — genuine multi-domain manifold
- 15 min ingest vs ~45 min — significantly faster pipeline
- H³ geometry with Möbius sharding — principled architecture vs ad-hoc placement
- Weighted chunk ranking — better concept descriptions when extraction is correct
- Fully in Rust — no Python dependency at runtime (currently)

---

## Root Cause & Fix

**Root cause:** n-gram extraction does not understand named entities. `q1 financial results (2026)` is extracted by spaCy as a time expression + noun phrase. Pilar's extractor sees `q1`, `financial`, `results`, `2026` as separate unigrams and doesn't surface the compound.

**Fix:** Hybrid extraction via PyO3 + spaCy:
- Keep sliding-window n-gram extractor for domain compounds (`starlink subscriber arpu`, `philosopher's stone`, `shares of class`) — spaCy NER misses these
- Add spaCy NER via PyO3 for named entities (ORG, PERSON, GPE, PRODUCT, EVENT, DATE, MONEY)
- Union both result sets before TF-IDF scoring

This requires Python as a runtime dependency. This is acceptable — PyO3 embeds Python as a library call, not a subprocess. One clean dependency (`pyo3` crate + `en_core_web_sm`) is preferable to maintaining a parallel Python service.

---

## Performance Baseline

| Metric | AscendingLLMa | Pilar |
|--------|--------------|-------|
| Ingest time (226 concepts) | ~45 min | ~15 min |
| Inference time | ~5 sec | ~2 sec |
| Corpora | 1 | 3 |
| Architecture | Ad-hoc Python | H³ Poincaré ball, Rust |
| Extraction | spaCy NER | n-gram sliding window |
| Retrieval quality | Better (this run) | Geometry sound, extraction gap |

---

## Next Steps

1. **PyO3 + spaCy hybrid extraction** — blocking issue for thesis experiment
2. **Model speed investigation** — gemma3:4b as middle ground between tinyllama and mistral
3. **Three-way thesis experiment** — raw vs manifold vs wrong context — after extraction fixed
4. **pilar-server** — HTTP layer after thesis experiment validated

---
