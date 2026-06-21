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

/// What ner.rs + tfidf.rs + embed.rs will eventually produce together —
/// a scored term with its embedding attached. Defined here, not upstream,
/// because those modules don't exist yet; this is placement's side of the
/// contract, so there's something concrete to build toward.
pub struct EmbeddedTerm {
    pub term: String,
    pub tfidf_score: f64,
    pub embedding: Vec<f64>,
    pub source_path: PathBuf,
    pub source_line: Option<usize>,
}

// ── Config ────────────────────────────────────────────────────────────────────

pub struct PlacementConfig {
    pub k_neighbors: usize,
    pub tfidf_threshold: f64,
    /// Fixed seed for the random projections below. "Random" only means
    /// "arbitrary and fixed" here — same seed, same projections, every run.
    pub projection_seed: u64,
    pub sharding: ShardingConfig,
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            tfidf_threshold: 0.001,
            projection_seed: 42,
            sharding: ShardingConfig::default(),
        }
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

/// Concepts grouped by where they ended up. `periphery` is keyed by
/// shard_id rather than a magic "root" string sitting in the same map —
/// km.rs writes `root` to one file and each `periphery` entry to its own.
pub struct PlacementResult {
    pub root: Vec<Concept>,
    pub periphery: HashMap<String, Vec<Concept>>,
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
    /// Radius comes from TF-IDF strength elsewhere, not from this; this
    /// function only answers "which way," never "how far."
    pub fn hyperbolic_direction(&self, embedding: &[f64]) -> [f64; 3] {
        let raw = [
            Self::dot(&self.hyperbolic[0], embedding),
            Self::dot(&self.hyperbolic[1], embedding),
            Self::dot(&self.hyperbolic[2], embedding),
        ];
        let norm = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
        if norm < 1e-10 {
            // Vanishingly unlikely with a real embedding — fall back to a
            // fixed axis rather than dividing by ~zero.
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
    let terms: Vec<EmbeddedTerm> = terms
        .into_iter()
        .filter(|t| t.tfidf_score >= config.tfidf_threshold)
        .collect();

    let mut result = PlacementResult {
        root: Vec::new(),
        periphery: HashMap::new(),
    };

    if terms.is_empty() {
        return result;
    }

    let max_score = terms.iter().map(|t| t.tfidf_score).fold(0.0_f64, f64::max);
    let embedding_dim = terms[0].embedding.len();
    let projections = Projections::new(embedding_dim, config.projection_seed);

    let embeddings: Vec<Vec<f64>> = terms.iter().map(|t| t.embedding.clone()).collect();
    let d = distance_matrix(&embeddings);
    let n = terms.len();
    let k = config.k_neighbors.min(n.saturating_sub(1));

    for (i, term) in terms.iter().enumerate() {
        let strength = (term.tfidf_score / max_score).min(1.0);

        // k nearest neighbors by embedding distance
        let mut neighbors: Vec<(usize, f64)> = (0..n).filter(|&j| j != i).map(|j| (j, d[(i, j)])).collect();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        neighbors.truncate(k);

        let neighborhood_indices: Vec<usize> =
            std::iter::once(i).chain(neighbors.iter().map(|(j, _)| *j)).collect();

        let sub_n = neighborhood_indices.len();
        let mut sub_d = DMatrix::zeros(sub_n, sub_n);
        for (a, &ia) in neighborhood_indices.iter().enumerate() {
            for (b, &ib) in neighborhood_indices.iter().enumerate() {
                sub_d[(a, b)] = d[(ia, ib)];
            }
        }

        // Both gromov_delta and eigenvalue_signature degrade gracefully on
        // tiny neighborhoods (empty quadruple loops, near-zero eigenvalue
        // totals) — no special-casing needed for isolated concepts.
        let delta = gromov_delta(&sub_d);
        let eigen: EigenSignature = eigenvalue_signature(&sub_d);

        // Low delta strongly suggests hyperbolic regardless of eigenvalue
        // signature — same tiebreaker as before, now living here as a
        // policy decision over two independent pieces of evidence, not
        // buried inside either geometry.rs function.
        let class = if delta < 1.0 { GeometryClass::Hyperbolic } else { eigen.class };

        let confidence = GeometryConfidence {
            gromov_delta: delta,
            eigenvalue_ratio: eigen.eigenvalue_ratio,
            first_dominance: eigen.first_dominance,
            neg_eigenvalue_fraction: eigen.neg_eigenvalue_fraction,
        };

        let coordinate = match class {
            GeometryClass::Hyperbolic => {
                let dir = projections.hyperbolic_direction(&term.embedding);
                let r = (1.0 - 0.9 * strength).min(0.95);
                ManifoldCoord::Hyperbolic {
                    position: [dir[0] * r, dir[1] * r, dir[2] * r],
                }
            }
            GeometryClass::Spherical => ManifoldCoord::Spherical {
                theta: projections.spherical_angle(&term.embedding),
            },
            GeometryClass::Flat => ManifoldCoord::Flat {
                position: vec![projections.flat_position(&term.embedding)],
            },
        };

        let mut concept = Concept {
            raw_term: term.term.clone(),
            label: String::new(),
            description: String::new(),
            coordinate,
            confidence,
            embedding: term.embedding.clone(),
            tfidf_score: term.tfidf_score,
            source_path: term.source_path.clone(),
            source_line: term.source_line,
        };

        // Sharding only applies to hyperbolic concepts — S¹ has no
        // boundary to fall toward, and ℝ¹ doesn't share the same
        // precision-loss failure mode that motivated periphery shards
        // in the first place.
        if let ManifoldCoord::Hyperbolic { position } = &concept.coordinate {
            let radius = (position[0] * position[0] + position[1] * position[1] + position[2] * position[2]).sqrt();

            if radius > config.sharding.periphery_radius {
                let assignment = registry.route(position, &config.sharding);
                let (shard_id, local_position) = match assignment {
                    ShardAssignment::Joined { shard_id, local_position }
                    | ShardAssignment::NewShard { shard_id, local_position } => (shard_id, local_position),
                };
                concept.coordinate = ManifoldCoord::Hyperbolic {
                    position: [local_position[0], local_position[1], local_position[2]],
                };
                result.periphery.entry(shard_id).or_default().push(concept);
                continue;
            }
        }

        result.root.push(concept);
    }

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_term(term: &str, tfidf: f64, embedding: Vec<f64>) -> EmbeddedTerm {
        EmbeddedTerm {
            term: term.into(),
            tfidf_score: tfidf,
            embedding,
            source_path: PathBuf::from("test.md"),
            source_line: None,
        }
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
        assert!(result.root.is_empty());
        assert!(result.periphery.is_empty());
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
        let all: Vec<&Concept> = result.root.iter().chain(result.periphery.values().flatten()).collect();
        assert!(all.iter().all(|c| c.raw_term != "noise"));
    }

    #[test]
    fn test_high_strength_lands_nearer_origin() {
        let mut registry = ShardRegistry::new();
        let config = PlacementConfig::default();
        let terms = vec![
            mock_term("strong", 0.9, vec![0.1, 0.2, 0.3, 0.4, 0.5]),
            mock_term("weak", 0.1, vec![0.4, 0.1, 0.5, 0.2, 0.3]),
        ];
        let result = place(terms, &config, &mut registry);

        let radius_of = |c: &Concept| match &c.coordinate {
            ManifoldCoord::Hyperbolic { position } => {
                (position[0].powi(2) + position[1].powi(2) + position[2].powi(2)).sqrt()
            }
            _ => panic!("expected hyperbolic for this test's tiny neighborhood"),
        };

        let strong = result.root.iter().find(|c| c.raw_term == "strong").unwrap();
        let weak = result.root.iter().find(|c| c.raw_term == "weak").unwrap();
        assert!(radius_of(strong) < radius_of(weak));
    }

    #[test]
    fn test_description_and_label_start_empty() {
        let mut registry = ShardRegistry::new();
        let config = PlacementConfig::default();
        let terms = vec![mock_term("manifold", 0.9, vec![0.1, 0.2, 0.3])];
        let result = place(terms, &config, &mut registry);
        let c = &result.root[0];
        assert_eq!(c.label, "");
        assert_eq!(c.description, "");
    }
}