# Pilar — Thesis

## The Problem

Large language models hallucinate. Small language models hallucinate more. But
the failure mode isn't randomness — it's *ungrounded* generation. A model given
no context invents context. A model given the right context can reason coherently
even at 24M parameters, as demonstrated by smaLLM: a character-level transformer
trained exclusively on Les Misérables that produced incoherent output in isolation
but became a consistent character when given structured memory and period-appropriate
linguistic framing.

The insight from smaLLM wasn't that small models are capable of more than expected.
It was that the *interface* to the model matters as much as the model itself. The
geometry of the context window determines the quality of the output.

## The Hypothesis

If you can represent a knowledge domain as a geometric structure — one where
semantic proximity corresponds to spatial proximity — then at query time you can
retrieve only the concepts geometrically nearest to the query, inject those as
context, and give even a small model enough grounding to reason coherently about
a specific topic without inventing facts from outside the domain.

The manifold is not a retrieval system bolted onto an LLM. It is the interface
layer that determines what the LLM sees.

## Why Hyperbolic Space

Natural language has hierarchical structure. Words cluster into topics, topics
cluster into domains, domains cluster into corpora. This is tree-like structure,
and tree-like structure embeds most naturally in hyperbolic space — specifically
the Poincaré ball, where the exponential growth of volume away from the origin
mirrors the exponential branching of hierarchical data.

High-strength concepts (those recurring frequently and distinctively across a
corpus) sit near the origin — central, dense, well-reinforced. Low-strength
concepts drift toward the boundary. As more corpora are ingested, the manifold's
effective centroid shifts to reflect the dominant semantic mass of the accumulated
knowledge. No shard is permanently "root" — centrality is emergent, not assigned.

Temporal and cyclical concepts (dates, recurring events, financial cycles) exhibit
ring structure and land naturally in S¹. Linear gradient concepts land in ℝ. The
product manifold H³ × S¹ × ℝ accommodates all three structural signatures without
forcing everything into a single geometry.

## What Pilar Does

Pilar ingests text corpora and builds a persistent knowledge manifold:

1. **Ingest** — chunk text, extract n-gram candidates, score by TF-IDF per corpus
2. **Embed** — embed each candidate term via nomic-embed-text
3. **Classify** — assign geometry (H³, S¹, or ℝ) via eigenvalue signature of the
   local neighborhood distance matrix, with Gromov delta as a tiebreaker for
   unambiguously tree-like neighborhoods
4. **Place** — project to a coordinate in the assigned geometry; radius in H³
   determined by normalized strength (stronger = closer to origin)
5. **Shard** — route each concept to the nearest shard anchor via Poincaré
   distance; spawn new shards when nothing is close enough
6. **Enrich** — summarize each concept from its most signal-dense source chunks
   (tinyllama), name it (mistral)
7. **Persist** — write shard-N.km files and registry.km

At query time:

1. Embed the query
2. Project to H³ using the same fixed random projections as ingest (seed=42)
3. Find nearest shard anchors via registry
4. Load only those shards (lazy)
5. Rank all concepts in loaded shards by Poincaré distance to query
6. Inject top-k concept descriptions as context
7. Ask the LLM to answer using only the injected facts

## What We've Shown

Three corpora — SpaceX S-1, Mackay's Extraordinary Popular Delusions, Brandenburg's
Wall Street speculation pamphlet — ingested into a single manifold produced:

- 226 concepts across 30 shards
- Meaningful geometric separation: Brandenburg's Wall Street vocabulary drifted to
  the manifold boundary (outer shards), while SpaceX and Mackay concepts coexist
  near the origin because despite different domains they share enough semantic
  overlap (institutions, power, financial mania, historical cycles)
- `starlink` classifying as Spherical — correct, it's a recurring service concept
  with ring structure — while `spacex` classified as Hyperbolic — correct, it's
  a hierarchy anchor
- Lazy loading working: a query about SpaceX revenue hit shard-14, shard-6, and
  shard-0 — not all 30 shards
- Nearest-neighbor retrieval returning geometrically coherent results: SpaceX
  queries return SpaceX concepts, alchemy queries return Mackay historical concepts

The manifold is queryable. The geometry is meaningful. The lazy loading works.

## What's Left

- **Query projection quality** — currently using strength=0.5 as a neutral query
  position. A better approach searches more shards or uses the query embedding
  direction more precisely.
- **Prompt construction** — inject ranked concepts as context and actually call
  the LLM to generate an answer. The mechanical retrieval works; the generation
  loop is unbuilt.
- **pilar-server** — HTTP layer over the inference loop for external consumption.
- **Multi-ingestion** — adding a new corpus to an existing manifold without
  re-ingesting everything. The registry and Möbius translation infrastructure
  support this; the pipeline doesn't wire it yet.
- **Vacuum step** — consolidating subsuming concepts (`wall` + `street` →
  `wall street`), pruning decayed concepts, merging geometrically proximate
  near-duplicates. Not ingestion's job; a maintenance pass over the manifold.
- **Projection seed in registry** — store seed and embedding dim in registry.km
  so the server doesn't have to assume them. Currently both are hardcoded
  (seed=42, dim=768 from nomic-embed-text) and deterministic, but that's an
  assumption that should be explicit.

## The Bet

A 7B parameter model given the right 5 sentences of context will outperform a
70B parameter model given no context on a specific domain question. Pilar is the
system that finds those 5 sentences geometrically, from a persistent manifold built
from real source material, without hallucinating facts that aren't there.

The manifold doesn't make the model smarter. It makes the model's context honest.