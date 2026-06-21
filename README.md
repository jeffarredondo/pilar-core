# pilar-core

Rust port of AscendingLLMa — a knowledge system where geometry *is* the
knowledge. Concepts get placed on a manifold based on their semantic
structure; querying is navigation, not search. The LLM is the voice; the
manifold is what it actually knows.

## Core idea

Concepts live on **M = H³ × S¹ × ℝ¹** — hyperbolic, spherical, and flat
components — chosen per-concept by classifying its local neighborhood
structure, not assigned up front. A small model handles ingestion-time
extraction and summarization; a larger model handles naming and final
synthesis. No fine-tuning, no RAG — the manifold is the only retrieval
mechanism.

Note: we need a thesis.md for this that goes into resonance, AL, and this paper https://arxiv.org/pdf/2601.02744 to discuss what our learnings have been and why an approach like this could carry water. We also need to benchmark it somehow on more than just the spacex s1. We need to build a corpus that will make our model more informed on "things" it couldn't know as part of its training. 

## Status

**Built and tested:**
- [x] `geometry.rs` — N-D Poincaré distance, Gromov delta, eigenvalue
      classification, Möbius addition / recentering
- [x] `sharding.rs` — routes periphery concepts to existing or new shards
- [x] `types.rs` — `Concept`, `ManifoldCoord`, `GeometryConfidence`
- [x] `placement.rs` — classifies geometry, projects embeddings to
      coordinates, hands periphery concepts to sharding
- [x] `ner.rs` — RAKE + capitalization extraction (replaces spaCy)
- [x] `tfidf.rs` — corpus-wide term scoring
- [x] `embed.rs` — Ollama embedding client (placement needs this)
- [x] `ingest.rs` — chunking (tfidf needs this)
- [x] Enrichment — small-model description, larger-model naming
      (`Concept.description` / `.label` are empty until this exists)
- [x] `km.rs` — shard file read/write
- [x] `pipeline.rs` — orchestration + timing instrumentation per stage

## Pipeline order

1. `ingest.rs` *(not built)* — chunk source text
2. `ner.rs` — extract candidate terms per chunk
3. `tfidf.rs` — score terms, full corpus, every run (cheap — see below)
4. `embed.rs` *(not built)* — embed each unique term via Ollama
5. `placement.rs` — classify geometry, project to coordinates, route
   periphery concepts via `sharding.rs`
6. Enrichment *(not built)* — small model writes `description`, larger
   model writes `label`
7. `km.rs` *(not built)* — persist shards to disk
8. `pipeline.rs` *(not built)* — runs 1–7 in order, timed per stage

## Design decisions worth not relitigating

- **M = H³ × S¹ × ℝ¹, not S².** S² has pole degeneracy (latitude isn't
  periodic); S¹ doesn't.
- **`periphery_radius` is reused for both** "has this concept left the
  root shard" and "is it close enough to join an existing periphery
  shard." No principled reason for two numbers — constant-curvature
  space has no privileged radius, this is a file-size knob, not a
  geometry knob. `0.9` is fine.
- **No `cluster_id`.** KMeans needs a global refit on every corpus
  update, which conflicts with incremental ingestion by construction.
  `shard_id` already serves as the coarse lookup.
- **`raw_term` vs `label`, deliberately split.** `raw_term` is
  deterministic (pure text extraction); `label` is the LLM's name for
  it — probabilistic, expected to vary run to run, not a bug to fix.
- **Direction comes from the embedding for all three geometries**
  (fixed seeded Gaussian random projections — generated once, reused
  forever). **Radius (H³ only) comes from TF-IDF strength** — H³ has a
  free radial axis to spend confidence on; S¹ and ℝ¹ don't.
- **Möbius recentering (`translate_to_origin`)**, not "just use
  `float64`," fixes precision loss near the H³ boundary. Confirmed
  against the κ-stereographic model's independent distance formula —
  see `geometry::tests::test_distance_matches_mobius_formula`.
  Reference: [andbloch.github.io/K-Stereographic-Model](https://andbloch.github.io/K-Stereographic-Model/)
- **Geometry classification runs on raw embedding-space k-NN
  distance, not `poincare_distance`.** Has to happen before a concept
  has hyperbolic coordinates at all — also avoids the circularity of
  testing "is this hyperbolic" using a metric that assumes it is.
- **TF-IDF and radius/confidence are fully recomputed every run** —
  cheap (pure counting + arithmetic, no network calls), so there's
  nothing to cache and no staleness to manage. Only embedding and
  enrichment (the genuinely expensive, LLM-backed steps) are worth
  caching per `raw_term` — not built yet.
- **`geometry.rs`, `sharding.rs`, `types.rs`, `placement.rs`,
  `tfidf.rs` are all pure** — no I/O, no network, fully testable
  without touching disk or Ollama.

## Known gaps / deferred (not bugs, just not built)

- `ShardRegistry` always starts empty — needs a `load()` path to
  resume anchors across runs.
- `placement.rs`'s k-NN search only sees the current batch, not
  previously-placed concepts — fine for a single run, wrong for
  incremental ingestion.
- `ner.rs` returns `HashSet<String>` (presence only) — discards
  intra-chunk frequency, so TF == DF whenever a term doesn't repeat
  within one chunk. Real gap, unmeasured impact.
- Flat/linear confidence metric (is ℝ¹ actually enough, or are
  unrelated concepts collapsing onto the same point) — deferred until
  real `Flat` concepts exist to test it against.
- TF-IDF pipe-delimited score dump — deferred, no consumer yet.
- `n = 1` for ℝⁿ — config, not architecture. Revisit if/when flat
  concepts show semantically unrelated terms landing suspiciously
  close together.

## Corpus note

Current test corpus: SpaceX S-1 + ~100-year-old Gutenberg
finance/markets texts. Heavy on hierarchy and recurring cycles by
nature — may fully explain why `Flat` never activated in earlier runs
(rather than a classifier calibration issue). Untested against a
genuinely linear, non-cyclical corpus (e.g. a strict changelog) —
worth trying if `Flat` stays empty once real ingestion runs.

## Dependencies so far

`nalgebra` (eigendecomposition), `rand` + `rand_distr` (seeded
projections), `rake` + `stop-words` (extraction). `toml` and an Ollama
client will be needed once `km.rs` / `embed.rs` exist.