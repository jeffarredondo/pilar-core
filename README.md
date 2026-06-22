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

Note: we still need a `thesis.md` that goes into Resonance, AscendingLLMa,
and this paper — https://arxiv.org/pdf/2601.02744 — to discuss what our
learnings have been and why an approach like this could carry water. Still
deliberately deferred (see Status) — today added real run data but also
real open questions (naming reliability, stopword leakage), which is the
opposite of stable. We also still need a benchmark corpus beyond SpaceX
S-1 — something that informs the model on "things" it couldn't know from
training, ideally including a genuinely non-cyclical text to actually test
whether `Flat` ever activates.

## Status

**Built and tested** (`cargo test` clean as of this update):
- [x] `geometry.rs` — N-D Poincaré distance, Gromov delta, eigenvalue
      classification, Möbius addition / recentering
- [x] `sharding.rs` — routes periphery concepts to existing or new shards;
      `ShardRegistry::load()` now resumes anchors across runs (was a gap,
      closed this session)
- [x] `types.rs` — `Concept`, `ManifoldCoord`, `GeometryConfidence`, all
      serde-derived for round-tripping through `km.rs`
- [x] `placement.rs` — classifies geometry, projects embeddings to
      coordinates, hands periphery concepts to sharding, caps output at
      `max_concepts`
- [x] `ner.rs` — RAKE + capitalization extraction (replaces spaCy)
- [x] `tfidf.rs` — corpus-wide term scoring, with a `min_occurrences`
      floor before IDF is even computed
- [x] `embed.rs` — Ollama `/api/embed` client
- [x] `ingest.rs` — chunking, character-indexed (not byte-indexed —
      matters for non-ASCII source text)
- [x] `enrich.rs` — small-model description (tinyllama), larger-model
      naming (mistral), via Ollama `/api/chat`
- [x] `km.rs` — shard + registry read/write via TOML
- [x] `pipeline.rs` — orchestration, timing instrumentation per stage,
      live progress bar during enrichment
- [x] `main.rs` — CLI binary (`pilar-core <files> [--output <dir>]
      [--dry-run]`)

**Deliberately not started:** `thesis.md` (see note above).

## Pipeline order

1. `ingest.rs` — chunk source text (character-indexed sliding window)
2. `ner.rs` — extract candidate terms per chunk (RAKE + capitalization)
3. `tfidf.rs` — score terms across the full corpus, filtered by
   `min_occurrences` before IDF runs at all
4. `embed.rs` — embed each surviving term via Ollama (`nomic-embed-text`)
5. `placement.rs` — classify geometry, project to coordinates, filter by
   `strength_threshold`, cap at `max_concepts`, route periphery concepts
   via `sharding.rs`
6. `enrich.rs` — tinyllama writes `description`, mistral writes `label`
7. `km.rs` — persist shards + registry snapshot to disk
8. `pipeline.rs` — runs 1–7 in order, timed per stage; `main.rs` is the
   entry point that actually calls it

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
  `km.rs` keys shards by `raw_term`, not `label`, for the same reason:
  `label` can collide across concepts (Python's own prototype needed a
  disambiguation suffix for exactly this), `raw_term` is unique within a
  run by construction.
- **Direction comes from the embedding for all three geometries**
  (fixed seeded Gaussian random projections — generated once, reused
  forever). **Radius (H³ only) comes from strength** — H³ has a free
  radial axis to spend confidence on; S¹ and ℝ¹ don't. `strength` is
  deliberately source-agnostic (today it's TF-IDF-derived via
  `normalize_to_strength`, but nothing in `placement.rs` assumes that —
  see the access-decay note below).
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
- **`min_occurrences` (in `tfidf.rs`, default `10`) is the real noise
  filter; `max_concepts` (in `placement.rs`, default `100`) is a
  separate, later scope decision — order matters and they are not
  interchangeable.** A real 64KB-vs-relative-strength-ratio run showed a
  ratio threshold doesn't behave predictably on small/quantized corpora
  (strength is max-normalized *within* a corpus, so the same ratio means
  wildly different survivor counts at different corpus sizes — see
  "First real run" below). A count cap applied *instead of* a real floor
  is worse, not better: it discards real signal and noise identically,
  with no way to tell them apart. `min_occurrences` measures something
  real (did this term actually recur), ported directly from the Python
  prototype's `min_tf=10`. `max_concepts` only truncates an *already*
  recurrence-filtered set — a resource/scope decision (enrichment costs
  ~2 Ollama calls per concept; uncapped real corpora produced 1000+
  surviving terms, hours of enrichment) mirroring Python's own
  `min_tf` → `n_top` order, which we'd initially missed and then found
  by rereading `ingest.py` directly.
- **Ingestion scores what's in the text, honestly — it does not
  editorialize on what counts as "real" content.** Project Gutenberg's
  license boilerplate scored #1 in the first real run (dense, repeated,
  concentrated in a couple of chunks — exactly TF-IDF's reward
  signature). That's correct, not a bug to filter around: deciding
  something is unimportant *despite* being structurally dense is an
  access/decay-mechanism's job (see below), not ingestion's. If this
  comes up again, the fix is "feed it different bytes," not "make
  ingestion smarter about what's real."
- **`geometry.rs`, `sharding.rs`, `types.rs`, `placement.rs`,
  `tfidf.rs` are all pure** — no I/O, no network, fully testable
  without touching disk or Ollama. `embed.rs` and `enrich.rs` each split
  pure logic (testable) from the actual HTTP call (not testable without
  a live Ollama) the same way. `pipeline.rs` inherits this same line at
  the orchestration level — `ingest_and_score`, `place_corpus`, and
  `write_all` are tested; `embed_terms`, `enrich_all`, and `run` aren't,
  on purpose.

## First real run (Brandenburg, 64KB, single file)

`min_occurrences: 10` took the term count from **1747 → 13**. Full
pipeline (ingest through write) completed in **39.6s total**, almost
entirely enrichment (39.1s — confirms the original timing-instrumentation
bet that LLM calls, not NER/chunking, would dominate). Result: 13 root
concepts, **0 periphery shards** — every concept's strength, max-normalized
within a pool of only 13 already-curated survivors, stayed comfortably
under the `0.9` periphery cutoff. Open question whether that's a
small-corpus artifact or a real property — untested against Mackay (936
chunks, 1859 terms post-floor) or SpaceX S-1 (825 chunks, 1120 terms
post-floor) with a *real* (non-dry-run) pass, since both have far more
internal strength spread to potentially push something past the cutoff.

Reading the actual 13 concepts surfaced two more findings:

- **`'the'` survived as a top-13 concept.** A bare stopword clearing
  `min_occurrences` is a real `ner.rs` gap — RAKE and/or the capitalized-
  phrase path is letting at least one stopword through. Not a tuning
  question, a correctness one. Not yet fixed.
- **Naming/summarization quality is real but not yet diagnosed cleanly.**
  Several `tinyllama` descriptions came back as a bare `"?"` instead of
  the explicitly-instructed fallback sentence — the guardrail prompt is
  correct, tinyllama just doesn't reliably follow it. Several `mistral`
  labels were garbled (`"rainy-sell-spree"`, `"panicprofitshield"`).
  Manually replaying the *exact* real prompts for both a `"?"`-description
  concept and a substantive-description concept through `ollama run
  mistral` directly produced **clean output both times** — and neither
  matched the original garbled output. That rules out "bad model pull"
  (confirmed via `ollama list`, normal 4.4GB mistral) and complicates
  "garbage-in/garbage-out" as the full explanation. Likely cause:
  **nothing currently sets a sampling temperature/seed on these Ollama
  calls**, so every call — identical prompt or not — is an independent
  stochastic draw with no reproducibility guarantee. This reframes the
  open question from "is mistral bad at naming" to "nothing here is
  deterministic enough to evaluate that yet." Unresolved, deliberately
  deferred — see `handoff.md`.

## Known gaps / deferred (not bugs, just not built)

- **`ner.rs` lets at least one stopword (`'the'`) through to scoring.**
  New finding from the first real run, not yet investigated or fixed.
- **No temperature/seed control on `enrich.rs`'s Ollama chat calls.**
  Makes single-sample output quality genuinely hard to evaluate, since
  identical inputs aren't guaranteed identical (or even similar) outputs.
  Worth fixing before drawing firm conclusions about either model's
  naming/summarization reliability.
- **`placement.rs`'s k-NN search only sees the current batch, not
  previously-placed concepts** — fine for a single run, wrong for
  incremental ingestion.
- **`ner.rs` returns `HashSet<String>` (presence only)** — discards
  intra-chunk frequency, so TF == DF whenever a term doesn't repeat
  within one chunk. Real gap, unmeasured impact.
- **Flat/linear confidence metric** — still untested against real
  `Flat` concepts; Brandenburg produced none (0 periphery shards
  entirely, let alone Flat specifically). Whether `Flat` ever activates
  on Mackay/SpaceX at full scale is still an open, real question, not a
  hypothetical one anymore.
- **TF-IDF pipe-delimited score dump** — deferred, no consumer yet.
- **`n = 1` for ℝⁿ** — config, not architecture. Revisit if/when flat
  concepts show semantically unrelated terms landing suspiciously
  close together.
- **`min_occurrences: 10` and `max_concepts: 100` are both still
  somewhat provisional**, even though the *mechanism and ordering* are
  settled (see Design decisions). `min_occurrences` ported Python's
  number directly without independently re-deriving it; `max_concepts`
  was chosen from "Python's n_top=60 felt low" rather than a principled
  target. Both reasonable starting points, neither rigorously tuned.
- **Embedding/enrichment caching keyed by `raw_term`** — still not
  built. Every run currently recomputes everything from scratch.

## Real numbers, for reference

| source | chunks | terms (post `min_occurrences: 10`) |
|---|---|---|
| Brandenburg (64KB) | 36 | 13 |
| Mackay (`Extraordinary Popular Delusions`) | 936 | 1859 |
| SpaceX S-1 | 825 | 1120 |

Mackay/SpaceX numbers are from `--dry-run` (scoring only, no embed/place/
enrich) — neither has had a real end-to-end pass yet.

## Dependencies

`nalgebra` (eigendecomposition), `rand` + `rand_distr` (seeded
projections), `rake` + `stop-words` (extraction), `reqwest` (blocking +
json, for Ollama HTTP calls), `serde` + `serde_json` (round-tripping
`Concept`/`ManifoldCoord`/etc.), `toml` (shard/registry persistence).
`main.rs` adds no new dependencies — just `std`.

## usage
`cargo run --release -- /Users/pocoloco/Documents/pilar-core/docs/brandenburg.txt --output ./km_output`
