# First Inference Test — Results

## Purpose: 

Validate that the manifold is queryable end-to-end — embed a query, project to H³, find nearest shards via the registry, compute Poincaré distances against loaded concepts, and return ranked results. No generation yet, purely mechanical retrieval.

## What we saw:

The SpaceX query worked well — tight distances (0.45-0.73), returning spacex, customers, segment, capacity — all legitimate SpaceX S-1 concepts. The geometry found the right neighborhood.
The alchemy query partially worked — believed, manner, duke are genuine Mackay historical concepts that co-occur with alchemical content. The raw_terms are correct even if the tinyllama-generated labels are hallucinated SpaceX descriptions. This is the enrichment quality problem, not a retrieval problem.
The witchcraft and Wall Street queries both pulled SpaceX concepts — revealing the main gap: query projection is too coarse. Using strength=0.5 as a neutral radius puts all queries at the same depth in H³ regardless of content, which biases retrieval toward whatever dominates the manifold at that radius (currently SpaceX, the largest corpus). The shard selection was geometrically reasonable but the fixed radius hurt concept-level ranking.

## Key finding: 

Retrieval works. The failure mode is query projection, not manifold structure. Fix the projection approach before building the generation loop.

### Resulting output from manifold

Registry loaded: 30 shards

Query: What is SpaceX's revenue and launch strategy?
────────────────────────────────────────────────────────────
Nearest shards: shard-14 (0.668), shard-6 (1.072), shard-0 (1.237)
Top 5 nearest concepts:
  [0.4538] trial -> "financial notes & disclosures"
           The accompanying notes are an integral part of these consolidated financial statements and contain i...
  [0.4824] customers -> "orbital scalar pioneer"
           The company aims to become a global leader in orbits and superior cost efficiency at scale, by enhan...
  [0.5813] spacex -> "spacex"
           SpaceX is advancing the boundaries of space technology and human spaceflight, developing reusable ro...
  [0.6469] segment -> "space investment shift"
           Space (the term for both the International Space Station and any other space-related activities) is ...
  [0.7354] capacity -> "satelliteexpansionboost"
           In increasing satellite capacity, STS has focused on launching higher-throughput satellite systems d...

Query: Tell me about witchcraft trials in Europe
────────────────────────────────────────────────────────────
Nearest shards: shard-0 (1.237), shard-14 (1.357), shard-8 (1.614)
Top 5 nearest concepts:
  [1.0478] continue -> "spacecominnovate"
           The company's business focuses on developing, manufacturing, and selling a variety of products relat...
  [1.0851] called -> "grogghexpansionfunding"
           The Company anticipates generating $17 billion in net proceeds from its upcoming initial public offe...
  [1.1104] capacity -> "satelliteexpansionboost"
           In increasing satellite capacity, STS has focused on launching higher-throughput satellite systems d...
  [1.2090] 2025 -> "ai mobile revolution"
           Intel has developed a new AI chip to power next-generation smartphones and laptops. The chip, called...
  [1.2368] continued -> "sustainable tech expansion"
           Our strategy is focused on driving sustainable revenue growth and expanding our margin through techn...

Query: Wall Street speculation and stock market panic
────────────────────────────────────────────────────────────
Nearest shards: shard-6 (1.035), shard-0 (1.237), shard-12 (1.299)
Top 5 nearest concepts:
  [0.5582] subject -> "space regulatory expansion"
           Space-based connectivity provider Starlink is subject to extensive federal procurement and regulator...
  [0.8428] march -> "march"
           In March 2026, SpaceX achieved success in deploying its reusable rocket and satellite manufacturing ...
  [0.8961] customers -> "orbital scalar pioneer"
           The company aims to become a global leader in orbits and superior cost efficiency at scale, by enhan...
  [0.9213] space -> "space tech revolution"
           Space is an industry that encompasses various technologies and applications across different markets...
  [0.9604] satellites -> "techbreakthroughtrajectory"
           The company's financial results for the three months ended March 31, 2026 (comprising payload delive...

Query: philosopher's stone and alchemy
────────────────────────────────────────────────────────────
Nearest shards: shard-13 (0.813), shard-12 (0.968), shard-0 (1.237)
Top 5 nearest concepts:
  [0.4973] outstanding -> "multisolarite pioneers"
           The SpaceX mission is to build systems and technologies necessary for making life multisolarite-leve...
  [0.5499] believed -> "regulatory spacex infrastructure"
           The company SpaceX is building the integrated hardware and software infrastructure of the future for...
  [0.5528] public -> "ai infrastructure growth"
           The company intends to use approximately $74.4 billion of net proceeds from this offering (or $85.7 ...
  [0.5541] manner -> "sustainableexpansionefficiency"
           The main challenge facing SpaceX is the sustainable and profitable revenue growth and expanding mark...
  [0.6303] duke -> "space propulsion evolution"
           The purpose of Duke's mission is to develop a new generation of advanced propulsion technology and s...

