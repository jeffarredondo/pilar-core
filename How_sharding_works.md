# How Pilar Finds Things Without Knowing What They Are

## The Surprising Part

The registry that routes queries to the right knowledge neighborhood contains
no words, no concept labels, no embeddings. It's just 30 coordinate triples
in three-dimensional space. And yet when you ask about SpaceX revenue, it sends
you to the SpaceX cluster. When you ask about alchemy, it sends you to the
Mackay historical cluster.

This isn't magic. It's geometry being consistent end to end.

---

## The Space We're Working In

Pilar uses the **Poincaré ball** — a model of hyperbolic space where the entire
infinite hyperbolic plane is compressed into the interior of a unit ball (radius < 1).

The key property: distance expands exponentially toward the boundary. Two points
near the center that are 1 unit apart in Euclidean space might be 10 units apart
in hyperbolic distance. Two points near the edge that look 1 unit apart in
Euclidean space might be 1000 units apart in hyperbolic distance.

The hyperbolic distance between two points **u** and **v** on the Poincaré ball is:

$$d(u, v) = \cosh^{-1}\left(1 + \frac{2\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}\right)$$

This geometry naturally fits hierarchical, tree-like data — the kind that language
produces. High-frequency, high-signal concepts sit near the origin (dense,
well-reinforced). Rare or domain-specific concepts drift toward the boundary.

---

## Placing Concepts

Every concept extracted from a text corpus gets an embedding from a language
model (nomic-embed-text, 768 dimensions). That embedding captures semantic
meaning as a point in high-dimensional space.

To place a concept on the 3D Poincaré ball, we use **fixed random projections** —
three random vectors **w₁, w₂, w₃** generated once with a fixed seed (42):

$$\text{direction} = \text{normalize}\left(\begin{bmatrix} w_1 \cdot e \\ w_2 \cdot e \\ w_3 \cdot e \end{bmatrix}\right)$$

where **e** is the concept's embedding and · is the dot product. This gives a
unit direction in H³. The radius is determined by the concept's strength
(how frequently and distinctively it appears in the corpus):

$$r = 1 - 0.9 \times \text{strength}$$

So a concept with strength 1.0 lands at radius 0.1 (near the origin, central).
A concept with strength 0.3 lands at radius 0.73 (further out). The final
position is:

$$\text{position} = r \times \text{direction}$$

Two concepts whose embeddings point in similar directions (semantically similar)
land near each other in H³. This is the key: **semantic similarity becomes
spatial proximity**.

---

## What a Shard Is

As concepts are placed, those that land far from the current cluster of concepts
(Poincaré distance > 0.9 from any existing anchor) become the anchor of a new
shard. Those that land close to an existing anchor join that shard.

Each shard is just a file (`shard-N.km`) containing the concepts in that
neighborhood, stored in their **local coordinates** — Möbius-translated so the
shard's anchor becomes the local origin:

$$\text{local}(x) = (-a) \oplus x$$

where **a** is the anchor position and ⊕ is Möbius addition (the hyperbolic
analog of vector addition). This keeps coordinates well-conditioned regardless
of where the shard sits in the global ball.

---

## What the Registry Is

The registry is simply a list of (shard_id, global_position) pairs — one anchor
per shard. For a manifold with 30 shards it looks like:

```
shard-0  →  [0.000,  0.000,  0.000]
shard-1  →  [-0.723, 0.458, -0.412]
shard-6  →  [-0.388, 0.213,  0.041]
...
```

That's it. No words. No concept names. Just coordinates in H³.

The anchor positions emerged from the data — shard-6's anchor ended up at
`[-0.388, 0.213, 0.041]` because that's where the Mackay alchemical concepts
happened to cluster geometrically during ingestion. The registry didn't decide
that. The geometry did.

---

## Querying

At query time:

1. **Embed the query** — same language model, same 768-dimensional space
2. **Project to H³** — same fixed random projections (seed 42), same formula
3. **Find nearest shards** — compute Poincaré distance from the query position
   to each anchor in the registry, return the closest K
4. **Load those shards** — read only the relevant `shard-N.km` files from disk
5. **Rank concepts** — compute Poincaré distance from query position to every
   concept in the loaded shards, return the closest

The query "What is SpaceX's revenue and launch strategy?" goes through the same
embedding model and the same random projections as the concepts extracted from
the SpaceX S-1 during ingestion. So it lands near the same region of H³ that
SpaceX concepts landed in — not because anyone told it to, but because the math
is consistent.

The registry routes it there without knowing what SpaceX is. It just knows
"there's a cluster at approximately this coordinate" and "this query is closest
to that cluster."

---

## Why This Works

Three properties make this possible:

**1. The same projection is used at ingest and query time.**
Fixed seed (42) means the random vectors **w₁, w₂, w₃** are identical every
run. A concept and a query about the same topic will project to similar directions
because they have similar embeddings, and similar directions in the same projection
become similar positions in H³.

**2. Hyperbolic distance respects semantic structure.**
The Poincaré ball's exponential boundary expansion means semantically central,
high-frequency concepts (near the origin) act as hubs, while domain-specific
concepts cluster at the boundary. Queries naturally land near the right cluster
because the geometry preserves the topology of the embedding space.

**3. The registry only needs to know where clusters are, not what's in them.**
Spatial indexing is a solved problem. The registry is essentially a K-nearest-
neighbor lookup in H³ over 30 points — trivially fast, requiring no knowledge
of concept content. The content lives in the shards. The registry just knows
the map.

---

## In Practice

Three corpora — a SpaceX S-1 filing, Mackay's *Extraordinary Popular Delusions*
(19th century financial manias and witchcraft trials), and a Wall Street
speculation pamphlet — produced 226 concepts across 30 shards. Query time for
nearest-shard lookup + concept ranking across loaded shards: **~2 seconds**,
dominated by the single embedding call. The Poincaré distance computations
themselves are microseconds.

The registry worked on the first try. No tuning, no training, no labeled data.
Just geometry being consistent.