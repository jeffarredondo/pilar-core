use serde::{Deserialize, Serialize};

use crate::geometry::{poincare_distance, translate_to_origin};

// ── Config ────────────────────────────────────────────────────────────────────

/// Controls how concepts are assigned to shards. `shard_radius` is the
/// maximum Poincaré distance from a shard anchor for a concept to join
/// that shard rather than spawning a new one. This is a file-size and
/// query-locality knob, not a geometry knob — constant curvature space
/// has no privileged radius.
pub struct ShardingConfig {
    pub shard_radius: f64,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self { shard_radius: 0.9 }
    }
}

// ── Types ─────────────────────────────────────────────────────────────────────

/// A shard's anchor point in the global H³ coordinate space. Every shard
/// has one — including shard-0, whose anchor is the origin by convention.
/// No shard is privileged: shard-0 just happens to be the first one
/// created. The centroid of the manifold shifts as more corpora are
/// ingested and the dominant semantic mass changes.
///
/// `position` is in global H³ coordinates, not relative to any other
/// shard. Local coordinates within a shard are computed by
/// translate_to_origin(anchor, concept_position) at ingestion time and
/// stored on the concept itself — the anchor is only needed for routing
/// and spatial indexing, not for distance computation within a shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAnchor {
    pub shard_id: String,
    pub position: Vec<f64>,
}

/// What `route` decided for a given concept. Either way, `local_position`
/// is where the concept lives in that shard's own recentered coordinate
/// frame — Möbius-translated so the shard anchor is the local origin.
/// This keeps distance computation well-conditioned regardless of where
/// the shard sits in global H³.
#[derive(Debug, Clone)]
pub enum ShardAssignment {
    Joined {
        shard_id: String,
        local_position: Vec<f64>,
    },
    NewShard {
        shard_id: String,
        local_position: Vec<f64>,
    },
}

/// Full spatial index over all shards — the "map" of the knowledge
/// manifold. Every shard is registered here, including shard-0.
///
/// At ingestion time: routes each concept to the nearest shard, spawning
/// a new shard when nothing is close enough.
///
/// At query time: `nearest_shards` finds which shards to load for a
/// given query coordinate, enabling lazy loading without scanning every
/// concept on disk.
///
/// The registry deliberately knows nothing about file paths or concept
/// content — it's purely a spatial index over anchor positions.
pub struct ShardRegistry {
    anchors: Vec<ShardAnchor>,
    next_id: usize,
}

impl ShardRegistry {
    /// Creates a new registry with shard-0 pre-registered at the origin.
    /// shard-0 is not semantically special — it's just where the first
    /// concepts land. As more corpora are ingested, the manifold's
    /// effective centroid shifts, and shard-0 may end up anywhere
    /// relative to the dominant semantic clusters.
    pub fn new() -> Self {
        let origin_anchor = ShardAnchor {
            shard_id: "shard-0".to_string(),
            position: vec![0.0, 0.0, 0.0],
        };
        Self {
            anchors: vec![origin_anchor],
            next_id: 1,
        }
    }

    /// Routes a concept to the nearest shard anchor. If the nearest
    /// anchor is within `shard_radius`, the concept joins that shard.
    /// Otherwise it becomes the anchor of a new shard.
    ///
    /// `global_position` is the concept's position in global H³
    /// coordinates, as computed by placement.rs before any recentering.
    pub fn route(&mut self, global_position: &[f64], config: &ShardingConfig) -> ShardAssignment {
        let nearest = self
            .anchors
            .iter()
            .map(|a| (a, poincare_distance(&a.position, global_position)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if let Some((anchor, dist)) = nearest {
            if dist <= config.shard_radius {
                let local_position = translate_to_origin(&anchor.position, global_position);
                return ShardAssignment::Joined {
                    shard_id: anchor.shard_id.clone(),
                    local_position,
                };
            }
        }

        // Nothing close enough — this concept becomes the anchor of a
        // new shard. Its local position is the origin of its own frame:
        // translate_to_origin(x, x) == [0,0,0] by definition.
        let shard_id = format!("shard-{}", self.next_id);
        self.next_id += 1;

        let local_position = translate_to_origin(global_position, global_position);

        self.anchors.push(ShardAnchor {
            shard_id: shard_id.clone(),
            position: global_position.to_vec(),
        });

        ShardAssignment::NewShard {
            shard_id,
            local_position,
        }
    }

    /// Returns the `top_k` nearest shard anchors to a global H³ position,
    /// sorted by ascending Poincaré distance. This is the entry point for
    /// lazy loading at query time: project a query embedding to a global
    /// coordinate, call this to find which shards to load, then compute
    /// exact distances only against concepts in those shards.
    ///
    /// Always includes at least one result (the nearest shard) as long
    /// as the registry is non-empty. `top_k` is clamped to the number
    /// of registered shards.
    pub fn nearest_shards(&self, global_position: &[f64], top_k: usize) -> Vec<(&ShardAnchor, f64)> {
        let mut distances: Vec<(&ShardAnchor, f64)> = self
            .anchors
            .iter()
            .map(|a| (a, poincare_distance(&a.position, global_position)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(top_k);
        distances
    }

    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    pub fn anchors(&self) -> &[ShardAnchor] {
        &self.anchors
    }

    pub fn next_id(&self) -> usize {
        self.next_id
    }

    /// Reconstructs a registry from persisted state. `next_id` is taken
    /// explicitly — see km.rs for why deriving it from shard ID suffixes
    /// would be fragile.
    pub fn load(anchors: Vec<ShardAnchor>, next_id: usize) -> Self {
        Self { anchors, next_id }
    }
}

impl Default for ShardRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn test_new_registry_has_shard_zero_at_origin() {
        let registry = ShardRegistry::new();
        assert_eq!(registry.anchor_count(), 1);
        let anchor = &registry.anchors()[0];
        assert_eq!(anchor.shard_id, "shard-0");
        assert_eq!(anchor.position, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_concept_near_origin_joins_shard_zero() {
        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();

        // A concept close to the origin should join shard-0.
        let assignment = registry.route(&[0.1, 0.0, 0.0], &config);

        match assignment {
            ShardAssignment::Joined { shard_id, .. } => {
                assert_eq!(shard_id, "shard-0");
            }
            ShardAssignment::NewShard { .. } => panic!("expected to join shard-0"),
        }
        assert_eq!(registry.anchor_count(), 1, "joining should not grow the registry");
    }

    #[test]
    fn test_concept_far_from_all_anchors_spawns_new_shard() {
        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();

        // A concept far from the origin spawns a new shard.
        let assignment = registry.route(&[0.92, 0.0, 0.0], &config);

        match assignment {
            ShardAssignment::NewShard { shard_id, local_position } => {
                assert_eq!(shard_id, "shard-1");
                assert!(norm(&local_position) < 1e-9, "anchor of new shard lands at its own origin");
            }
            ShardAssignment::Joined { .. } => panic!("expected a new shard"),
        }
        assert_eq!(registry.anchor_count(), 2);
    }

    #[test]
    fn test_nearby_concept_joins_existing_non_origin_shard() {
        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();

        // Spawn shard-1 far from origin.
        registry.route(&[0.92, 0.0, 0.0], &config);

        // A concept nearby shard-1 should join it, not spawn shard-2.
        let assignment = registry.route(&[0.93, 0.01, 0.0], &config);

        match assignment {
            ShardAssignment::Joined { shard_id, .. } => {
                assert_eq!(shard_id, "shard-1");
            }
            ShardAssignment::NewShard { .. } => panic!("expected to join shard-1"),
        }
        assert_eq!(registry.anchor_count(), 2, "joining should not grow the registry");
    }

    #[test]
    fn test_distant_concepts_spawn_separate_shards() {
        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();

        registry.route(&[0.92, 0.0, 0.0], &config);  // shard-1
        let assignment = registry.route(&[-0.92, 0.0, 0.0], &config); // too far from shard-0 and shard-1

        match assignment {
            ShardAssignment::NewShard { shard_id, .. } => assert_eq!(shard_id, "shard-2"),
            ShardAssignment::Joined { .. } => panic!("expected a new shard"),
        }
        assert_eq!(registry.anchor_count(), 3);
    }

    #[test]
    fn test_loaded_registry_continues_id_sequence_without_collision() {
        let existing_anchors = vec![
            ShardAnchor { shard_id: "shard-0".to_string(), position: vec![0.0, 0.0, 0.0] },
            ShardAnchor { shard_id: "shard-1".to_string(), position: vec![0.92, 0.0, 0.0] },
            ShardAnchor { shard_id: "shard-2".to_string(), position: vec![-0.92, 0.0, 0.0] },
        ];
        let mut registry = ShardRegistry::load(existing_anchors, 3);
        let config = ShardingConfig::default();

        let assignment = registry.route(&[0.0, 0.92, 0.0], &config);

        match assignment {
            ShardAssignment::NewShard { shard_id, .. } => {
                assert_eq!(shard_id, "shard-3", "should continue from next_id=3");
            }
            ShardAssignment::Joined { .. } => panic!("expected a new shard"),
        }
    }

    #[test]
    fn test_nearest_shards_returns_sorted_by_distance() {
        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();

        registry.route(&[0.92, 0.0, 0.0], &config);   // shard-1
        registry.route(&[-0.92, 0.0, 0.0], &config);  // shard-2

        // Query near shard-1 — it should come back first.
        let nearest = registry.nearest_shards(&[0.85, 0.0, 0.0], 3);

        assert_eq!(nearest.len(), 3);
        assert_eq!(nearest[0].0.shard_id, "shard-1");
        // Distances should be ascending.
        assert!(nearest[0].1 <= nearest[1].1);
        assert!(nearest[1].1 <= nearest[2].1);
    }

    #[test]
    fn test_nearest_shards_clamps_to_available_anchors() {
        let registry = ShardRegistry::new(); // only shard-0
        let nearest = registry.nearest_shards(&[0.1, 0.0, 0.0], 10);
        assert_eq!(nearest.len(), 1, "can't return more shards than exist");
    }

    #[test]
    fn test_join_picks_nearest_of_multiple_anchors() {
        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();

        registry.route(&[0.92, 0.0, 0.0], &config);   // shard-1
        registry.route(&[-0.92, 0.0, 0.0], &config);  // shard-2

        // Closer to shard-2 than shard-1.
        let assignment = registry.route(&[-0.90, 0.02, 0.0], &config);

        match assignment {
            ShardAssignment::Joined { shard_id, .. } => assert_eq!(shard_id, "shard-2"),
            ShardAssignment::NewShard { .. } => panic!("expected a join"),
        }
        assert_eq!(registry.anchor_count(), 3);
    }

    #[test]
    fn test_accessors_expose_what_km_needs_to_persist() {
        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();
        registry.route(&[0.92, 0.0, 0.0], &config);

        assert_eq!(registry.anchors().len(), 2); // shard-0 + shard-1
        assert_eq!(registry.next_id(), 2);
    }
}