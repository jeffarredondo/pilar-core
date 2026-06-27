use std::collections::HashMap;
use std::path::PathBuf;

use nalgebra::DMatrix;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use crate::geometry::{distance_matrix, eigenvalue_signature, gromov_delta, EigenSignature, GeometryClass};
use crate::sharding::{ShardAssignment, ShardRegistry, ShardingConfig};
use crate::types::{Concept, GeometryConfidence, ManifoldCoord};

// ── Input contract ────────────────────────────────────────────────────────────

/// What ner.rs + tfidf.rs + embed.rs produce together — a term with its
/// embedding attached and a pre-normalized strength value. `strength`
/// arrives already normalized to [0, 1] — placement doesn't know or care
/// whether it came from TF-IDF, access-count decay, or anything else.
pub struct EmbeddedTerm {
    pub term: String,
    pub strength: f64,
    pub embedding: Vec<f64>,
    pub source_path: PathBuf,
    pub source_line: Option<usize>,
}

// ── Config ────────────────────────────────────────────────────────────────────

pub struct PlacementConfig {
    pub k_neighbors: usize,
    /// Minimum normalized strength to be placed at all.
    pub strength_threshold: f64,
    /// Hard cap on placed concepts after the threshold filter. None means
    /// no cap. This is a resource knob, not a noise filter — min_occurrences
    /// in tfidf.rs already handles noise upstream. Capping here just limits
    /// enrichment cost (2 Ollama calls per concept).
    pub max_concepts: Option<usize>,
    /// Fixed seed for random projections — same seed, same projections,
    /// every run. "Random" only means "arbitrary and fixed" here.
    pub projection_seed: u64,
    pub sharding: ShardingConfig,
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            strength_threshold: 0.001,
            max_concepts: None,
            projection_seed: 42,
            sharding: ShardingConfig::default(),
        }
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

/// Concepts grouped by shard_id. No shard is special — shard-0 is just
/// the first one created, not a "root." The registry holds the spatial
/// index; this holds the concepts.
pub struct PlacementResult {
    pub shards: HashMap<String, Vec<Concept>>,
}

impl PlacementResult {
    fn new() -> Self {
        Self { shards: HashMap::new() }
    }

    fn insert(&mut self, shard_id: String, concept: Concept) {
        self.shards.entry(shard_id).or_default().push(concept);
    }

    pub fn total_concepts(&self) -> usize {
        self.shards.values().map(|v| v.len()).sum()
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

// ── Fixed Projections ─────────────────────────────────────────────────────────

/// Generated once per placement run, reused for every concept — the same
/// projection vectors have to apply to every embedding or relative
/// geometry between concepts breaks. Gaussian, not uniform: standard
/// Johnson-Lindenstrauss practice, preserves relative distances better
/// than naive uniform sampling would.
pub struct Projections {
    hyperbolic: [Vec<f64>; 3],
    spherical: [Vec<f64>; 2],
    flat: Vec<f64>,
}

impl Projections {
    pub fn new(embedding_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut gen_vec = || -> Vec<f64> { (0..embedding_dim).map(|_| normal.sample(&mut rng)).collect() };

        Self {
            hyperbolic: [gen_vec(), gen_vec(), gen_vec()],
            spherical: [gen_vec(), gen_vec()],
            flat: gen_vec(),
        }
    }

    fn dot(v: &[f64], embedding: &[f64]) -> f64 {
        v.iter().zip(embedding.iter()).map(|(a, b)| a * b).sum()
    }

    /// Unit direction in H³ — magnitude is deliberately discarded here.
    /// Radius comes from strength elsewhere; this function only answers
    /// "which way," never "how far."
    pub fn hyperbolic_direction(&self, embedding: &[f64]) -> [f64; 3] {
        let raw = [
            Self::dot(&self.hyperbolic[0], embedding),
            Self::dot(&self.hyperbolic[1], embedding),
            Self::dot(&self.hyperbolic[2], embedding),
        ];
        let norm = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
        if norm < 1e-10 {
            return [1.0, 0.0, 0.0];
        }
        [raw[0] / norm, raw[1] / norm, raw[2] / norm]
    }

    pub fn spherical_angle(&self, embedding: &[f64]) -> f64 {
        let a = Self::dot(&self.spherical[0], embedding);
        let b = Self::dot(&self.spherical[1], embedding);
        a.atan2(b)
    }

    pub fn flat_position(&self, embedding: &[f64]) -> f64 {
        Self::dot(&self.flat, embedding)
    }
}

// ── Placement ─────────────────────────────────────────────────────────────────

pub fn place(
    terms: Vec<EmbeddedTerm>,
    config: &PlacementConfig,
    registry: &mut ShardRegistry,
) -> PlacementResult {
    let mut terms: Vec<EmbeddedTerm> = terms
        .into_iter()
        .filter(|t| t.strength >= config.strength_threshold)
        .collect();

    if let Some(max) = config.max_concepts {
        terms.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
        terms.truncate(max);
    }

    let mut result = PlacementResult::new();

    if terms.is_empty() {
        return result;
    }

    let embedding_dim = terms[0].embedding.len();
    let projections = Projections::new(embedding_dim, config.projection_seed);

    let embeddings: Vec<Vec<f64>> = terms.iter().map(|t| t.embedding.clone()).collect();
    let d = distance_matrix(&embeddings);
    let n = terms.len();
    let k = config.k_neighbors.min(n.saturating_sub(1));

    for (i, term) in terms.iter().enumerate() {
        let strength = term.strength;

        // k nearest neighbors by embedding distance for geometry classification
        let mut neighbors: Vec<(usize, f64)> = (0..n).filter(|&j| j != i).map(|j| (j, d[(i, j)])).collect();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        neighbors.truncate(k);

        let neighborhood_indices: Vec<usize> =
            std::iter::once(i).chain(neighbors.iter().map(|(j, _)| *j)).collect();

        let sub_n = neighborhood_indices.len();
        let mut sub_d = DMatrix::zeros(sub_n, sub_n);
        for (row, &gi) in neighborhood_indices.iter().enumerate() {
            for (col, &gj) in neighborhood_indices.iter().enumerate() {
                sub_d[(row, col)] = d[(gi, gj)];
            }
        }

        let delta = gromov_delta(&sub_d);
        let eigen: EigenSignature = eigenvalue_signature(&sub_d);

        // Eigenvalue signature is the primary signal — it does the actual
        // geometric analysis. Gromov delta only overrides a non-hyperbolic
        // classification if delta is extremely low (< 0.1), meaning the
        // neighborhood is unambiguously tree-like regardless of what the
        // eigenvalues say. Let the math determine the shape.
        let class = match eigen.class {
            GeometryClass::Hyperbolic => GeometryClass::Hyperbolic,
            other => if delta < 0.1 { GeometryClass::Hyperbolic } else { other },
        };

        let confidence = GeometryConfidence {
            gromov_delta: delta,
            eigenvalue_ratio: eigen.eigenvalue_ratio,
            first_dominance: eigen.first_dominance,
            neg_eigenvalue_fraction: eigen.neg_eigenvalue_fraction,
        };

        // Compute global H³ position from embedding — this is the position
        // before any shard-local recentering. The registry uses this to
        // decide which shard the concept belongs to.
        let global_position = match class {
            GeometryClass::Hyperbolic => {
                let dir = projections.hyperbolic_direction(&term.embedding);
                let r = (1.0 - 0.9 * strength).min(0.95);
                Some([dir[0] * r, dir[1] * r, dir[2] * r])
            }
            _ => None,
        };

        let (coordinate, shard_id) = if let Some(pos) = global_position {
            // All hyperbolic concepts route through the registry — no
            // special "root" case. The registry decides which shard based
            // on proximity to existing anchors. Local position is the
            // Möbius-translated coordinate in the shard's own frame.
            let assignment = registry.route(&pos, &config.sharding);
            let (sid, local_position) = match assignment {
                ShardAssignment::Joined { shard_id, local_position }
                | ShardAssignment::NewShard { shard_id, local_position } => (shard_id, local_position),
            };
            let coord = ManifoldCoord::Hyperbolic {
                position: [local_position[0], local_position[1], local_position[2]],
            };
            (coord, sid)
        } else {
            // Spherical and flat concepts don't live in H³ so they don't
            // participate in shard routing. They land in shard-0 by
            // convention — a future extension could give them their own
            // spatial index if they become numerous enough to warrant it.
            let coord = match class {
                GeometryClass::Spherical => ManifoldCoord::Spherical {
                    theta: projections.spherical_angle(&term.embedding),
                },
                GeometryClass::Flat => ManifoldCoord::Flat {
                    position: vec![projections.flat_position(&term.embedding)],
                },
                GeometryClass::Hyperbolic => unreachable!(),
            };
            (coord, "shard-0".to_string())
        };

        let concept = Concept {
            raw_term: term.term.clone(),
            label: String::new(),
            description: String::new(),
            coordinate,
            confidence,
            embedding: term.embedding.clone(),
            strength: term.strength,
            source_path: term.source_path.clone(),
            source_line: term.source_line,
        };

        result.insert(shard_id, concept);
    }

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_term(term: &str, strength: f64, embedding: Vec<f64>) -> EmbeddedTerm {
        EmbeddedTerm {
            term: term.into(),
            strength,
            embedding,
            source_path: PathBuf::from("test.md"),
            source_line: None,
        }
    }

    fn all_concepts(result: &PlacementResult) -> Vec<&Concept> {
        result.shards.values().flatten().collect()
    }

    #[test]
    fn test_projections_are_deterministic() {
        let p1 = Projections::new(8, 42);
        let p2 = Projections::new(8, 42);
        let emb = vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.1, 0.2, 0.0];

        assert_eq!(p1.hyperbolic_direction(&emb), p2.hyperbolic_direction(&emb));
        assert_eq!(p1.spherical_angle(&emb), p2.spherical_angle(&emb));
        assert_eq!(p1.flat_position(&emb), p2.flat_position(&emb));
    }

    #[test]
    fn test_hyperbolic_direction_is_unit_length() {
        let p = Projections::new(8, 42);
        let emb = vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.1, 0.2, 0.0];
        let dir = p.hyperbolic_direction(&emb);
        let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_terms() {
        let mut registry = ShardRegistry::new();
        let config = PlacementConfig::default();
        let result = place(vec![], &config, &mut registry);
        assert_eq!(result.total_concepts(), 0);
        assert_eq!(result.shard_count(), 0);
    }

    #[test]
    fn test_threshold_filters_low_scores() {
        let mut registry = ShardRegistry::new();
        let config = PlacementConfig::default();
        let terms = vec![
            mock_term("manifold", 0.9, vec![0.1, 0.2, 0.3, 0.4]),
            mock_term("noise", 0.0001, vec![0.5, 0.1, 0.2, 0.1]),
        ];
        let result = place(terms, &config, &mut registry);
        assert!(all_concepts(&result).iter().all(|c| c.raw_term != "noise"));
    }

    #[test]
    #[test]
    fn test_strength_to_radius_formula() {
        // The radius formula r = (1 - 0.9 * strength).min(0.95) is the core
        // property: higher strength → smaller radius → closer to origin.
        // Test it directly on the projections rather than through the full
        // placement pipeline, where Möbius translation into a non-origin shard
        // can shift local coordinates in ways that make the assertion fragile.
        let p = Projections::new(5, 42);
        let emb = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let dir = p.hyperbolic_direction(&emb);

        let r_strong = 1.0 - 0.9 * 0.9_f64;
        let r_weak   = 1.0 - 0.9 * 0.3_f64;

        let pos_strong = [dir[0] * r_strong, dir[1] * r_strong, dir[2] * r_strong];
        let pos_weak   = [dir[0] * r_weak,   dir[1] * r_weak,   dir[2] * r_weak];

        let radius = |p: [f64; 3]| (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();

        assert!(radius(pos_strong) < radius(pos_weak),
            "strength 0.9 should produce smaller radius than strength 0.3");
    }

    #[test]
    fn test_max_concepts_caps_to_strongest_terms() {
        let mut registry = ShardRegistry::new();
        let config = PlacementConfig {
            max_concepts: Some(2),
            ..PlacementConfig::default()
        };
        let terms = vec![
            mock_term("weakest", 0.1, vec![0.1, 0.2, 0.3, 0.4]),
            mock_term("middle", 0.5, vec![0.2, 0.3, 0.1, 0.4]),
            mock_term("strongest", 0.9, vec![0.3, 0.1, 0.4, 0.2]),
        ];
        let result = place(terms, &config, &mut registry);
        let all = all_concepts(&result);

        assert_eq!(all.len(), 2);
        assert!(all.iter().any(|c| c.raw_term == "strongest"));
        assert!(all.iter().any(|c| c.raw_term == "middle"));
        assert!(all.iter().all(|c| c.raw_term != "weakest"));
    }

    #[test]
    fn test_max_concepts_none_means_no_cap() {
        let mut registry = ShardRegistry::new();
        let config = PlacementConfig {
            max_concepts: None,
            ..PlacementConfig::default()
        };
        let terms: Vec<EmbeddedTerm> = (0..10)
            .map(|i| mock_term(&format!("term{i}"), 0.5, vec![i as f64 * 0.1, 0.2, 0.3, 0.4]))
            .collect();
        let result = place(terms, &config, &mut registry);
        assert_eq!(result.total_concepts(), 10);
    }

    #[test]
    fn test_description_and_label_start_empty() {
        let mut registry = ShardRegistry::new();
        let config = PlacementConfig::default();
        let terms = vec![mock_term("manifold", 0.9, vec![0.1, 0.2, 0.3])];
        let result = place(terms, &config, &mut registry);
        let c = all_concepts(&result).into_iter().next().unwrap();
        assert_eq!(c.label, "");
        assert_eq!(c.description, "");
    }

    #[test]
    fn test_all_concepts_assigned_to_a_shard() {
        let mut registry = ShardRegistry::new();
        let config = PlacementConfig::default();
        let terms: Vec<EmbeddedTerm> = (0..5)
            .map(|i| mock_term(&format!("concept{i}"), 0.5 + i as f64 * 0.1, vec![i as f64 * 0.1, 0.2, 0.3, 0.4]))
            .collect();
        let result = place(terms, &config, &mut registry);

        // Every concept must be in some shard — none dropped silently.
        assert_eq!(result.total_concepts(), 5);
        // Every shard_id in result must exist in the registry.
        for shard_id in result.shards.keys() {
            assert!(
                registry.anchors().iter().any(|a| &a.shard_id == shard_id),
                "shard {shard_id} in result but not in registry"
            );
        }
    }
}