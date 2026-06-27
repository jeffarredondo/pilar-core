# pilar-core

Rust implementation of a knowledge manifold system where geometry *is* the
knowledge. Concepts extracted from text corpora get placed on a product
manifold based on their semantic structure; querying is geometric navigation,
not keyword search. The LLM is the voice; the manifold is what it actually knows.

## Core idea

Concepts live on **M = H³ × S¹ × ℝ¹** — hyperbolic, spherical, and flat
components — chosen per-concept by classifying its local neighborhood
structure, not assigned up front. A small model handles ingestion-time
extraction and summarization; a larger model handles naming. No fine-tuning,
no RAG — the manifold is the only retrieval mechanism.

See `thesis.md` for the full argument, and `how_sharding_works.md` for an
accessible explanation of why the registry works without knowing what
concepts mean.

## Status

**Built and tested** (`cargo test` clean, 84 tests):
- [x] `geometry.rs` — N-D Poincaré distance, Gromov delta, eigenvalue
      classification, Möbius addition / recentering
- [x] `sharding.rs` — full spatial index over all shards; no root/periphery
      distinction; `ShardRegistry::new()` pre-registers shard-0 at origin;
      `nearest_shards(position, top_k)` for lazy loading at query time;
      `load()` resumes across runs without ID collision
- [x] `types.rs` — `Concept`, `ManifoldCoord`, `GeometryConfidence`, all
      serde-derived; `embedding` field has `#[serde(skip)]` — used at ingest
      time for placement, never written to disk
- [x] `placement.rs` — classifies geometry via eigenvalue signature (primary)
      + Gromov delta (tiebreaker at < 0.1); projects embeddings to coordinates;
      routes all concepts through registry; `PlacementResult` is a flat
      `HashMap<String, Vec<Concept>>` keyed by shard_id — no special root
- [x] `ner.rs` — sliding-window n-gram extraction (replaced RAKE +
      capitalized_phrases); edges-only rule: first and last token must be
      non-stopwords, internal stopwords allowed; `NerConfig { max_n: 3 }`
- [x] `tfidf.rs` — per-corpus term scoring with `min_occurrences` floor
      before IDF is computed; `normalize_to_strength` produces [0,1] signal
- [x] `embed.rs` — Ollama `/api/embed` client
- [x] `ingest.rs` — character-indexed sliding-window chunker (not byte-indexed
      — matters for non-ASCII source text)
- [x] `enrich.rs` — small-model description (tinyllama) with fixed prompt
      that avoids the fill-in-the-blank echo pattern; larger-model naming
      (mistral) via Ollama `/api/chat`
- [x] `km.rs` — shard + registry read/write via TOML; files named `shard-N.km`
      and `registry.km`; keyed by `raw_term`, not `label`
- [x] `pipeline.rs` — orchestration with timing per stage and live progress
      bar; `max_concepts_per_source` caps per corpus before pooling (not after);
      placement cap removed
- [x] `main.rs` — CLI binary with multi-file support and `--dry-run`
- [x] `src/bin/infer.rs` — inference binary; loads registry + shards, embeds
      query, projects to H³, finds nearest shards, ranks concepts by Poincaré
      distance; confirmed working against real `.km` output

## Pipeline order

1. `ingest.rs` — chunk source text (character-indexed sliding window)
2. `ner.rs` — extract n-gram candidates per chunk (edges-only stopword rule)
3. `tfidf.rs` — score terms per corpus, filtered by `min_occurrences`
4. `embed.rs` — embed each surviving term via Ollama (`nomic-embed-text`)
5. `placement.rs` — classify geometry, project to coordinates, route all
   concepts through `sharding.rs`
6. `enrich.rs` — tinyllama writes `description`, mistral writes `label`
7. `km.rs` — persist `shard-N.km` files + `registry.km`
8. `pipeline.rs` — orchestrates 1–7 timed; `main.rs` is the entry point

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

# Inference test against existing km_output
cargo run --bin infer --release
```

## Design decisions worth not relitigating

- **M = H³ × S¹ × ℝ¹, not S².** S² has pole degeneracy (latitude isn't
  periodic); S¹ doesn't.

- **No root/periphery distinction.** shard-0 is pre-registered at the
  origin `[0,0,0]` by convention, not privilege. The manifold's effective
  centroid is emergent from whichever corpora have been ingested — a corpus
  that dominates the current manifold defines "central." Add a large new
  corpus and the centroid shifts. This is correct behavior, not a bug.

- **All concepts route through the registry.** Previously concepts with
  radius < 0.9 went to "root" and concepts with radius ≥ 0.9 went to
  periphery shards. Now all concepts route through `ShardRegistry::route()`.
  The registry decides which shard based on Poincaré proximity to existing
  anchors. Shard-0's anchor is the origin so near-origin concepts naturally
  join it — same behavior, no special case.

- **Eigenvalue signature is the primary geometry classifier; Gromov delta
  is a tiebreaker only.** `delta < 0.1` forces Hyperbolic regardless of
  eigenvalue call — unambiguously tree-like neighborhoods. Previously
  `delta < 1.0` overrode everything, causing all concepts to land Hyperbolic.
  Now `starlink` → Spherical (ring structure, recurring service), `spacex`
  → Hyperbolic (hierarchy anchor). Let the math determine the shape.

- **Embeddings are not persisted.** `#[serde(skip)]` on `embedding: Vec<f64>`
  in `Concept`. Used during placement for geometry classification and
  coordinate projection — done at ingest time, never needed again. At query
  time, coordinates are sufficient for Poincaré distance. File sizes dropped
  from 400KB-1.5MB to 30-100KB per corpus.

- **Per-source capping, not global.** `max_concepts_per_source: Some(100)`
  in `PipelineConfig` caps each corpus independently before pooling.
  Three corpora at 100 each = up to 300 concepts in the manifold, not 100.
  Placement itself applies no additional cap (`max_concepts: None`).

- **Fixed seeded random projections (seed=42).** The same projection vectors
  must be used at ingest and query time or coordinates are in different spaces.
  Currently hardcoded at seed=42 and embedding_dim=768 (nomic-embed-text).
  **TODO:** store projection_seed and embedding_dim in `registry.km` so the
  server doesn't have to assume them.

- **`raw_term` vs `label`, deliberately split.** `raw_term` is deterministic
  (pure text extraction); `label` is the LLM's name — probabilistic, expected
  to vary run to run, not a bug to fix. `km.rs` keys shards by `raw_term`.

- **`min_occurrences` (default 10) is the noise filter; `max_concepts_per_source`
  (default 100) is a separate scope decision.** Order matters: floor first,
  cap second. A cap applied to an unfiltered pool is arbitrary; a cap applied
  after a recurrence floor is "how many of the good candidates can we afford."

- **Ingestion does not editorialize on what counts as real content.** If
  boilerplate scores high, that's TF-IDF being correct — boilerplate is dense
  and concentrated. Deciding something is unimportant despite being structurally
  dense is an access/decay mechanism's job, not ingestion's.

- **Overlap between n-grams of different lengths is intentional.** `spacex`,
  `spacex rocket`, and `rocket` can all survive as separate concepts.
  Consolidating subsuming concepts is a future vacuum step over the manifold,
  not ingestion's job. Same reasoning as the Gutenberg-ranking-#1 call: don't
  editorialize at extraction time, let density and geometry decide later.

- **`geometry.rs`, `sharding.rs`, `types.rs`, `placement.rs`, `tfidf.rs`
  are all pure** — no I/O, no network, fully testable without Ollama.
  `embed.rs` and `enrich.rs` each split pure logic (testable) from the HTTP
  call (not testable without live Ollama). `pipeline.rs` follows the same line.

## Real run results (3 corpora)

Three corpora ingested together in a single run:

| corpus | chunks | terms (post cap) |
|---|---|---|
| SpaceX S-1 | 825 | 100 |
| Mackay — *Extraordinary Popular Delusions* | 936 | 100 |
| Brandenburg — Wall Street pamphlet | 36 | 26 |
| **total** | **1797** | **226** |

Result: **226 concepts, 30 shards**, enrichment in ~7 minutes on M4 Mini.

Shard structure reflected genuine semantic clustering:
- **shard-0** (origin) — cross-domain concepts where SpaceX and Mackay
  overlap: `witchcraft`, `france`, `death`, `king` alongside `starlink`,
  `spacex`, `launch`, `satellite`
- **shard-6** — Mackay alchemical cluster: `philosopher's`, `alchymy`,
  `stone`, `gold`, `evil`, `secret`
- **shard-7** — mixed historical/legal: `europe`, `germany`, `london`,
  `shares of class`, `regulatory`
- **shards 1-5, 9-10, 17, 19-29** — Brandenburg Wall Street vocabulary
  at the manifold boundary: `sell`, `buying`, `profits`, `wall street`,
  `stocks`

Brandenburg's entire domain vocabulary drifted to the periphery because it's
semantically distant from both SpaceX and Mackay in embedding space. No shard
is permanently central — centrality is emergent from the current corpus mix.

Geometry classification produced meaningful results:
- `starlink` → **Spherical** (ring structure — recurring service offering)
- `spacex` → **Hyperbolic** (hierarchy anchor)
- `december`, `2023`, `2024` → **Spherical** (temporal/cyclical)
- Brandenburg concepts → **Hyperbolic** (single tight domain, all tree-like)

## First inference test

Query → embed → project to H³ → `nearest_shards` → load shards → rank by
Poincaré distance → return top-k concepts. **~2 seconds end to end**, dominated
by the single embedding call. Poincaré distance computations are microseconds.

SpaceX revenue query returned `spacex`, `customers`, `segment`, `capacity` at
distances 0.45–0.73. Alchemy query returned `believed`, `manner`, `duke` —
genuine Mackay historical concepts that co-occur with alchemical content.

Main gap: query projection uses a fixed `strength=0.5` radius, which biases
toward whatever dominates the manifold at that depth. Fix before building the
generation loop.

## Known gaps / deferred

- **Query projection quality.** Fixed `strength=0.5` at query time puts all
  queries at radius 0.55 regardless of content. Need to search more shards
  or use a better radius strategy. Try searching shard-0 always (cross-domain
  hub) plus nearest_shards from the registry.

- **Generation loop not built.** Retrieval works. Wire top-k concept
  descriptions to mistral as context and compare grounded vs ungrounded output.
  That's the thesis experiment.

- **Projection seed + embedding dim not stored in registry.** Currently
  hardcoded (seed=42, dim=768). Should be written to `registry.km` at ingest
  time and read at inference time.

- **pilar-server not started.** HTTP layer over the inference loop. After
  the generation loop proves out.

- **Iterative ingestion not wired.** Infrastructure supports it (registry
  persists, Möbius handles new anchors) but pipeline always starts fresh.
  Defer until inference is validated.

- **Vacuum step not built.** Consolidating subsuming n-grams (`wall` + `street`
  → `wall street`), pruning decayed concepts, merging near-duplicates. Future
  maintenance pass over the manifold.

- **Embedding/enrichment caching not built.** Every run recomputes from scratch.
  Fine at current corpus sizes.

- **`Flat` geometry never activated.** All real corpora produced only
  Hyperbolic and Spherical concepts. Need a genuinely linear/gradient corpus
  to test the ℝ¹ component.

- **Naming/summarization non-determinism is intentional.** No temperature/seed
  on Ollama chat calls — identical prompts may produce different labels across
  runs. This is acceptable: the coordinate is fixed (deterministic projection),
  the label is just a human-readable handle. Labels should orbit the same
  semantic concept across runs without being identical. Drift into a completely
  unrelated domain would indicate guardrail failure, not non-determinism.

- **Small-model guardrail failures are accepted, not chased.** tinyllama and
  mistral will sometimes ignore format instructions regardless of prompt
  engineering. The one specific fixable case (the fill-in-the-blank echo
  pattern) is fixed. General unreliability is accepted — the manifold provides
  grounding, the model provides generation, they're separate jobs.

## Dependencies

- `nalgebra` (eigendecomposition), 
- `rand` + `rand_distr` (seeded projections),
- `stop-words` (stopword set for n-gram extraction), 
- `reqwest` (blocking + json,
Ollama HTTP), 
- `serde` + `serde_json` (round-tripping `Concept`/`ManifoldCoord`),
- `toml` (shard/registry persistence).

## usage
```bash
# single corpus
cargo run --release -- /Users/pocoloco/Documents/pilar-core/docs/brandenburg.txt --output ./km_output
cargo run --release -- /Users/pocoloco/Documents/pilar-core/docs/spacex_s1.txt --output ./km_output

# multi-run 
cargo run --release -- \
  /Users/pocoloco/Documents/pilar-core/docs/spacex_s1.txt \
  /Users/pocoloco/Documents/pilar-core/docs/mackay.txt \
  /Users/pocoloco/Documents/pilar-core/docs/brandenburg.txt \
  --output ./km_output \
  --dry-run

# manifold test
cargo run --bin infer --release    

```