# Knowledge Manifold — Thesis

## The Problem

Large language models hallucinate. The failure mode isn't randomness — it's ungrounded generation. A model given no context invents context. A model given the right context can reason coherently even at small parameter counts, as demonstrated by smaLLM: a character-level transformer trained on Les Misérables that produced incoherent output in isolation but became consistent when given structured memory and period-appropriate framing.

The insight from smaLLM wasn't that small models are more capable than expected. It was that the interface to the model matters as much as the model itself. The geometry of the context window determines the quality of the output.

---

## The Hypothesis

If you can represent a knowledge domain as a geometric structure — one where semantic proximity corresponds to spatial proximity — then at query time you can retrieve only the concepts geometrically nearest to the query, inject those as context, and give even a small model enough grounding to reason coherently about a specific topic without inventing facts from outside the domain.

The manifold is not a retrieval system bolted onto an LLM. It is the interface layer that determines what the LLM sees.

---

## Why Hyperbolic Space

Natural language has hierarchical structure. Words cluster into topics, topics into domains, domains into corpora. This is tree-like structure, and tree-like structure embeds most naturally in hyperbolic space — specifically the Poincaré ball, where exponential volume growth away from the origin mirrors exponential branching of hierarchical data (Nickel & Kiela, 2017).

High-strength concepts sit near the origin — central, dense, well-reinforced. Low-strength concepts drift toward the boundary. As more corpora are ingested, the manifold's effective centroid shifts to reflect the dominant semantic mass of accumulated knowledge. No shard is permanently "root" — centrality is emergent, not assigned.

Key property from Nickel & Kiela: even 2D hyperbolic space outperforms 200D Euclidean space for hierarchical data. H³ gives us three dimensions for multiple co-existing domain hierarchies (SpaceX, Mackay, Brandenburg coexisting in one manifold) while remaining parsimonious.

---

## Geometry Assumptions

We initially hypothesized a product manifold M = H³ × S¹ × ℝ¹, accommodating hierarchical, cyclical, and linear structure respectively. Empirical testing across three heterogeneous corpora showed S¹ and ℝ¹ activating only as artifacts of classification thresholds, not genuine structural signals. We simplify to **M = H³**.

This is an empirical finding, not a limitation. The Poincaré ball already accommodates what S¹ and ℝ¹ were intended to capture:

- Cyclical structure, if present in sufficient data, emerges as angular clustering in H³ rather than requiring a separate S¹ component
- Linear/gradient structure appears as radial gradients at varying depths in H³
- Concepts with ambiguous neighborhood structure land at high radius near the boundary, where local geometry is nearly flat
- Sharding handles sparse peripheral regions organically — no separate geometry needed

**Honest caveats:**
- Geometry classification is local (k=5 neighborhood) — an empirical bet, not a proof
- Flat geometry (ℝ¹) has never activated on real data — open question whether this reflects the assumption holding or a classification gap
- H³-only eliminates the cross-geometry distance problem entirely
- Higher dimensions (Hⁿ, n > 3) may be warranted for richer corpora; H³ is sufficient at current scale

**Methodological distinction from prior work:** Most mixed-curvature embedding work either assumes geometry up front or learns it end-to-end. Pilar infers geometry locally from neighborhood structure at ingestion time, per concept, without training. This is a real methodological difference worth documenting.

---

## What Pilar Does

1. **Ingest** — chunk text, extract n-gram candidates, score by TF-IDF per corpus
2. **Embed** — embed each surviving candidate via nomic-embed-text
3. **Classify** — assign geometry via eigenvalue signature of local neighborhood distance matrix; Gromov delta as tiebreaker for unambiguously tree-like neighborhoods
4. **Place** — project to H³ coordinate; radius from normalized strength (stronger = closer to origin); ambiguous concepts (delta ≥ 0.15) get radius penalty pushing toward boundary
5. **Shard** — route each concept to nearest shard anchor via Poincaré distance; spawn new shards when nothing is close enough
6. **Enrich** — summarize each concept from its most signal-dense source chunks; name it via LLM
7. **Persist** — write `shard-N.km` files and `registry.km`

**At query time:**

1. Embed the query
2. Project to H³ using same fixed random projections as ingest (seed=42)
3. Scale query position to interior of ball (0.5 radius) — query has no strength
4. Find nearest shard anchors via registry
5. Load only those shards (lazy)
6. Rank all concepts in loaded shards by Poincaré distance to query
7. Inject top-k concept descriptions as context
8. Ask LLM to answer using only the injected facts

---

## What We've Shown

Three corpora — SpaceX S-1 (filed May 20, 2026), Mackay's *Extraordinary Popular Delusions* (1841), Brandenburg's Wall Street speculation pamphlet (1800s) — ingested into a single manifold:

- 226 concepts, 32 shards
- Meaningful geometric separation: Brandenburg's Wall Street vocabulary drifted to the boundary; SpaceX and Mackay coexist near the origin (shared semantic mass: institutions, power, financial mania)
- Lazy loading working: SpaceX revenue query hit 6 shards, not all 32
- Generation loop working: mistral stays grounded to injected facts, correctly says "the facts do not contain this information" rather than hallucinating

**Comparison with AscendingLLMa (predecessor):**

AL demonstrated the core thesis on a single SpaceX corpus — a model with zero training data on a document filed 18 days prior correctly answered factual questions when given manifold context. Pilar extends this to multiple heterogeneous corpora.

Current gap: AL's spaCy NER extracted precise named entities (`q1 financial results (2026)`). Pilar's n-gram extractor produces generic terms (`result`, `2024`, `segment`). The retrieval geometry works; concept extraction quality is the remaining gap.

---

## The Bet

A 7B parameter model given the right 5 sentences of context will outperform a 70B parameter model given no context on a specific domain question. Pilar is the system that finds those 5 sentences geometrically, from a persistent manifold built from real source material, without hallucinating facts that aren't there.

The manifold doesn't make the model smarter. It makes the model's context honest.

---

## References

- Nickel, M. & Kiela, D. (2017). *Poincaré Embeddings for Learning Hierarchical Representations.* arXiv:1705.08039