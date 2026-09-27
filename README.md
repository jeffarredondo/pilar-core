# pilar-core

Rust implementation of a knowledge manifold system where geometry *is* the knowledge. Concepts extracted from text corpora get placed on a hyperbolic manifold based on their semantic structure. Querying is geometric navigation, not keyword search. The LLM is the voice; the manifold is what it actually knows.

---

## Core idea

Concepts live on **M = H³** — the Poincaré ball model of hyperbolic space. Geometry is assigned per-concept by classifying its local neighborhood structure at ingestion time, not learned end-to-end and not assumed up front. A small model handles description; a larger model handles naming. No fine-tuning, no RAG — the manifold is the only retrieval mechanism.

See `thesis.md` for the full argument.

---

## Status

**Built and tested** (`cargo test` clean):

- [x] `geometry.rs` — Poincaré distance, Gromov delta, eigenvalue classification, Möbius addition / recentering
- [x] `sharding.rs` — full spatial index; `ShardRegistry::new()` pre-registers shard-0 at origin; `nearest_shards(position, top_k)` for lazy loading; `load()` resumes across runs without ID collision
- [x] `types.rs` — `Concept`, `ManifoldCoord`, `GeometryConfidence`, all serde-derived; `embedding` field has `#[serde(skip)]` — used at ingest, never written to disk
- [x] `placement.rs` — classifies geometry via eigenvalue signature + Gromov delta; everything places into H³; ambiguous concepts (delta ≥ 0.15) get radius penalty toward boundary; routes all concepts through registry
- [x] `ngram.rs` (formerly `ner.rs`) — sliding-window n-gram extraction; edges-only rule: first and last token must be non-stopwords, internal stopwords allowed; `NgramConfig { max_n: 3 }`
- [x] `spacy_ner.rs` — real named-entity recognition via an embedded Python + spaCy (`en_core_web_sm`), loaded once via PyO3; whitelists ORG/PERSON/GPE/PRODUCT/EVENT/DATE/MONEY. Hardened against real filing text: whitespace/newline artifacts inside an entity are collapsed (`normalize_entity_text`), financial-table bleed-through producing multi-`$` MONEY entities is filtered out (`is_table_bleed_money`), and SEC-style parenthetical negatives (`$(1,943) million`, which spaCy strips the sign from) are repaired back to a signed figure (`repair_parenthetical_negative_money`). Also works around a macOS-specific PyO3 issue where the embedded interpreter fails to find the active venv's site-packages at runtime (`ensure_venv_site_packages_on_path`); not seen on Linux.
- [x] `extraction.rs` — hybrid extractor: unions `ngram.rs` (domain compounds spaCy has no label for) with `spacy_ner.rs` (precise entity anchors n-grams fragment); degrades to n-grams-only with a stderr warning if spaCy isn't installed
- [x] `tfidf.rs` — per-corpus term scoring with `min_occurrences` floor; `normalize_to_strength` produces [0,1] signal
- [x] `embed.rs` — Ollama `/api/embed` client. Terms are embedded with source context, not bare (`pipeline.rs::build_embedding_input` pairs each term with up to 300 chars of its source chunk) — a bare short/numeric term like `$4,694 million` carries too little semantic signal on its own to be retrieved by a natural-language query.
- [x] `ingest.rs` — character-indexed sliding-window chunker
- [x] `enrich.rs` — weighted chunk ranking by TF-IDF signal strength (not count); source-isolated chunk selection (no cross-corpus bleed); small-model description + larger-model naming via Ollama
- [x] `km.rs` — shard + registry read/write via TOML; `registry.km` stores `projection_seed` and `embedding_dim`
- [x] `pipeline.rs` — orchestration with timing per stage and live progress bar; `max_concepts_per_source` caps per corpus before pooling
- [x] `main.rs` — CLI binary with multi-file support and `--dry-run`
- [x] `src/bin/infer.rs` — inference binary; loads registry + shards, embeds query, projects to H³, finds nearest shards, ranks by Poincaré distance, injects top-5 as context, calls LLM for grounded answer. Full (untruncated) descriptions go into the RAG prompt; the old `.chars().take(100)` was truncating descriptions *before* they reached the prompt, not just at display — the LLM never saw anything past character 100. Truncation now happens only at print time.

---

## Pipeline

```
ingest → extraction (ngram + spacy_ner) → tfidf → embed → placement → sharding → enrich → km
```

1. `ingest.rs` — chunk source text (character-indexed sliding window, 2000 chars, 200 overlap)
2. `extraction.rs` — per chunk, union `ngram.rs`'s n-gram candidates (edges-only stopword rule, max_n=3) with `spacy_ner.rs`'s named entities
3. `tfidf.rs` — score terms per corpus, filter by `min_occurrences`
4. `embed.rs` — embed each surviving term via Ollama (`nomic-embed-text`)
5. `placement.rs` — classify geometry, project to H³ coordinates, route through `sharding.rs`
6. `enrich.rs` — write `description` and `label` per concept
7. `km.rs` — persist `shard-N.km` files + `registry.km`

---

## Running

```bash
# Single corpus
cargo run --release -- docs/spacex_s1.txt --output ./km_output

# Multiple corpora (per-source cap applied independently)
cargo run --release -- \
  docs/spacex_s1.txt \
  docs/mackay.txt \
  docs/brandenburg.txt \
  --output ./km_output

# Dry run — score only, no Ollama calls, no files written
cargo run --release -- docs/spacex_s1.txt --dry-run

# Inference + generation against existing km_output
cargo run --bin infer --release
```

Config lives in `pilar.toml` at project root (same level as `Cargo.toml`). Dry-run bypasses config and uses defaults.

---

## Design decisions

**M = H³, not M = H³ × S¹ × ℝ¹.** We tested the product manifold. S¹ and ℝ¹ activated only as artifacts of classification thresholds, not genuine structural signals. H³ is sufficient — cyclical and linear structure emerges as angular clustering and radial gradients within the ball. Sharding handles the periphery organically.

**Geometry is classified, not learned.** Per-concept, at ingest time, from local neighborhood structure. Eigenvalue signature of the distance matrix is the primary signal; Gromov delta is the tiebreaker. This is methodologically distinct from mixed-curvature learned embedding work.

**No root/periphery distinction.** Shard-0 is pre-registered at origin `[0,0,0]` by convention, not privilege. Centrality is emergent from corpus content — add a large new corpus and the effective centroid shifts. Correct behavior, not a bug.

**All concepts route through the registry.** The registry decides which shard based on Poincaré proximity to existing anchors. No special cases.

**Embeddings are not persisted.** `#[serde(skip)]` on `embedding: Vec<f64>`. Used during placement for geometry classification and coordinate projection — done at ingest time, never needed again. File sizes: ~30-100KB per corpus vs 400KB-1.5MB with embeddings.

**Per-source capping, not global.** `max_concepts_per_source: Some(100)` caps each corpus independently before pooling. Three corpora at 100 each = up to 300 concepts total.

**Fixed seeded projections (seed=42).** Same projection vectors at ingest and query time — required for coordinates to be in the same space. Seed and embedding dim stored in `registry.km`.

**Chunk ranking is strength-weighted, not count-based.** Co-occurring terms are weighted by their TF-IDF strength, not just counted. Prevents generic high-frequency terms from dominating chunk selection. Per-source filtering prevents cross-corpus signal bleed.

**`raw_term` vs `label`, deliberately split.** `raw_term` is deterministic (pure text extraction). `label` is the LLM's name — probabilistic, expected to vary run to run. `km.rs` keys shards by `raw_term`.

**Small-model hallucination is accepted, not chased.** The manifold provides grounding; the model provides generation. They're separate jobs.

---

## Real run results (3 corpora, June 28 2026)

| corpus | chunks | terms (post cap) |
|---|---|---|
| SpaceX S-1 | 825 | 100 |
| Mackay — *Extraordinary Popular Delusions* | 936 | 100 |
| Brandenburg — Wall Street pamphlet | 36 | 26 |
| **total** | **1797** | **226** |

Result: **226 concepts, 32 shards**

Shard structure reflected genuine semantic clustering. Brandenburg's domain vocabulary drifted to the manifold boundary (semantically distant from both SpaceX and Mackay). SpaceX and Mackay coexist near the origin — different domains, shared semantic mass around institutions, power, and financial systems.

---

## Known gaps

**Resolved since the June 28 handoff:** hybrid extraction (n-gram ∪ spaCy NER) is built and validated end-to-end against real corpora; spaCy MONEY-entity noise (whitespace bleed, table bleed, sign-stripped parenthetical negatives) is cleaned up; term embeddings now carry source-chunk context instead of being embedded bare; and the `infer.rs` prompt-truncation bug is fixed. See git history / session notes for details — not re-litigating any of it here.

- **N-gram-only noise still present** — the hybrid extractor's n-gram half still surfaces generic fragments (`result`, `segment`, `remained`, `4,694 million driven`) that TF-IDF has to filter after the fact. spaCy's entities (`elon musk`, `$4,694 million`) don't have this problem, but they only cover the labeled entity types — a frequency ceiling on the n-gram side is still worth revisiting.
- **Multi-corpus chunk ranking bug** — weighted ranking works on single corpus; cross-corpus strength normalization may dilute domain-specific terms in multi-corpus runs.
- **Model speed** — mistral 7B (~55 min for 226 concepts) is slow. gemma3:4b pulled locally, not yet tested.
- **Three-way thesis experiment** — not yet run cleanly. Now unblocked by the hybrid extractor — worth rerunning.
- **pilar-server** — HTTP layer not built yet.
- **Iterative ingestion** — infrastructure supports it, pipeline doesn't wire it.
- **Vacuum step** — consolidating subsuming n-grams, pruning decayed concepts. Future maintenance pass.

---

## Dependencies

`nalgebra` · `rand` + `rand_distr` · `stop-words` · `reqwest` · `serde` + `serde_json` · `toml` · `pyo3` (Python embedding for spaCy NER)

`pyo3` requires Python 3.8+ and `en_core_web_sm` in the runtime environment (the same interpreter `python3` resolves to at build time — set `PYO3_PYTHON` if you need to pin a specific one). Install with:

```bash
pip install spacy && python3 -m spacy download en_core_web_sm
```

If spaCy isn't installed, the pipeline still runs — `extraction.rs` logs a warning to stderr once and falls back to n-gram extraction only. This is a real runtime dependency for a research system, not a bug; the hybrid mode is strictly better when spaCy is present.

## References
[Poincaré Embeddings for
Learning Hierarchical Representations](https://arxiv.org/pdf/1705.08039)

## usage
```bash
# single corpus
cargo run --release --bin pilar-core -- /Users/pocoloco/Documents/pilar-core/docs/mackay.txt --output ./km_output
cargo run --release -- /Users/pocoloco/Documents/pilar-core/docs/spacex_s1.txt --output ./km_output

# multi-run 
cargo run --release --bin pilar-core -- \
  /Users/pocoloco/Documents/pilar-core/docs/spacex_s1.txt \
  /Users/pocoloco/Documents/pilar-core/docs/mackay.txt \
  /Users/pocoloco/Documents/pilar-core/docs/brandenburg.txt \
  --output ./km_output \
  --dry-run

# manifold test
cargo run --bin infer --release    

```