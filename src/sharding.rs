use serde::{Deserialize, Serialize};

use crate::geometry::{poincare_distance, translate_to_origin};

// ── Config ────────────────────────────────────────────────────────────────────

/// Reuses the same number as placement's periphery cutoff, per the earlier
/// call — there's no principled reason for "past the cutoff" and "close
/// enough to join an existing shard" to be different values. Constant
/// curvature space has no privileged radius; this is a file-size knob,
/// not a geometry knob.
pub struct ShardingConfig {
    pub periphery_radius: f64,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self { periphery_radius: 0.9 }
    }
}

// ── Types ─────────────────────────────────────────────────────────────────────

/// A known periphery shard, tracked only by where its anchor sits in ROOT
/// shard coordinates. This is the entire "map" — no separate disk, no
/// second coordinate system, just a sparse set of points in the same H³
/// every concept is already embedded in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAnchor {
    pub shard_id: String,
    pub root_position: Vec<f64>,
}

/// What `route` decided for a given concept. Either way, `local_position`
/// is where the concept lives in *that shard's own* recentered coordinates
/// — never the raw root-coordinate position, which is exactly what we're
/// trying to get away from once something's past the cutoff.
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

/// Owns the running list of periphery shard anchors for one ingestion run.
/// Persistence to disk (writing this out as part of `Meta`) is km.rs's job,
/// not this one — this is purely the in-memory routing decision.
pub struct ShardRegistry {
    anchors: Vec<ShardAnchor>,
    next_id: usize,
}

impl ShardRegistry {
    pub fn new() -> Self {
        Self {
            anchors: Vec::new(),
            next_id: 0,
        }
    }

    /// Routes a concept that placement.rs has *already* determined is past
    /// the periphery cutoff. This function doesn't re-check that — its only
    /// job is deciding which shard a past-cutoff concept belongs to.
    ///
    /// `root_position` must be the concept's position in ROOT shard
    /// coordinates (i.e. before any recentering has been applied).
    pub fn route(&mut self, root_position: &[f64], config: &ShardingConfig) -> ShardAssignment {
        let nearest = self
            .anchors
            .iter()
            .map(|a| (a, poincare_distance(&a.root_position, root_position)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if let Some((anchor, dist)) = nearest {
            if dist <= config.periphery_radius {
                let local_position = translate_to_origin(&anchor.root_position, root_position);
                return ShardAssignment::Joined {
                    shard_id: anchor.shard_id.clone(),
                    local_position,
                };
            }
        }

        // Nothing close enough — this concept becomes a new anchor.
        // translate_to_origin(self, self) is exactly the zero vector
        // (test_translate_self_to_zero in geometry.rs covers this property).
        let shard_id = format!("periphery-{}", self.next_id);
        self.next_id += 1;

        let local_position = translate_to_origin(root_position, root_position);

        self.anchors.push(ShardAnchor {
            shard_id: shard_id.clone(),
            root_position: root_position.to_vec(),
        });

        ShardAssignment::NewShard {
            shard_id,
            local_position,
        }
    }

    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    /// What km.rs needs to persist this registry's state to disk.
    pub fn anchors(&self) -> &[ShardAnchor] {
        &self.anchors
    }

    /// What km.rs needs to persist this registry's state to disk.
    pub fn next_id(&self) -> usize {
        self.next_id
    }

    /// Reconstructs a registry from previously-persisted state — the
    /// gap flagged since this file was first built: ShardRegistry::new()
    /// always started empty, so a resumed run could never find an
    /// existing periphery shard to join, only ever spawn fresh ones.
    ///
    /// next_id is taken explicitly rather than re-derived by parsing the
    /// numeric suffix off each anchor's shard_id — that would work today,
    /// but ties correctness to a naming convention staying exactly
    /// "periphery-{N}" forever. Persisting the counter directly doesn't
    /// care what the IDs happen to look like.
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
    fn test_first_concept_always_spawns() {
        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();

        let assignment = registry.route(&[0.92, 0.0, 0.0], &config);

        match assignment {
            ShardAssignment::NewShard { local_position, .. } => {
                assert!(norm(&local_position) < 1e-9, "first anchor should land at zero");
            }
            ShardAssignment::Joined { .. } => panic!("empty registry should never join"),
        }
        assert_eq!(registry.anchor_count(), 1);
    }

    #[test]
    fn test_nearby_concept_joins_existing_shard() {
        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();

        let first = registry.route(&[0.92, 0.0, 0.0], &config);
        let first_id = match first {
            ShardAssignment::NewShard { shard_id, .. } => shard_id,
            _ => unreachable!(),
        };

        // A point close to the first anchor (well within periphery_radius
        // of it) should join, not spawn a second shard.
        let second = registry.route(&[0.93, 0.01, 0.0], &config);

        match second {
            ShardAssignment::Joined { shard_id, local_position } => {
                assert_eq!(shard_id, first_id);
                // It joined a non-origin anchor, so local coords should have
                // actually moved, not be identical to the root position.
                assert!(norm(&local_position) < norm(&[0.93, 0.01, 0.0]) + 1.0);
            }
            ShardAssignment::NewShard { .. } => {
                panic!("expected this point to join the existing nearby shard")
            }
        }
        assert_eq!(registry.anchor_count(), 1, "joining shouldn't create a new anchor");
    }

    #[test]
    fn test_distant_concept_spawns_new_shard() {
        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();

        registry.route(&[0.92, 0.0, 0.0], &config);

        // Opposite side of the ball — definitely not within periphery_radius
        // of the first anchor.
        let second = registry.route(&[-0.92, 0.0, 0.0], &config);

        match second {
            ShardAssignment::NewShard { local_position, .. } => {
                assert!(norm(&local_position) < 1e-9);
            }
            ShardAssignment::Joined { .. } => {
                panic!("expected a far-away point to spawn its own shard")
            }
        }
        assert_eq!(registry.anchor_count(), 2);
    }

    #[test]
    fn test_loaded_registry_continues_id_sequence_without_collision() {
        // Simulates resuming a run where periphery-0 and periphery-1
        // already exist on disk. A naive load() that reset next_id to 0
        // would spawn a new "periphery-0", colliding with the real one.
        let existing_anchors = vec![
            ShardAnchor {
                shard_id: "periphery-0".to_string(),
                root_position: vec![0.92, 0.0, 0.0],
            },
            ShardAnchor {
                shard_id: "periphery-1".to_string(),
                root_position: vec![-0.92, 0.0, 0.0],
            },
        ];
        let mut registry = ShardRegistry::load(existing_anchors, 2);
        let config = ShardingConfig::default();

        // Far from both existing anchors -- should spawn fresh, not join.
        let assignment = registry.route(&[0.0, 0.92, 0.0], &config);

        match assignment {
            ShardAssignment::NewShard { shard_id, .. } => {
                assert_eq!(shard_id, "periphery-2", "should continue from next_id, not restart at 0");
            }
            ShardAssignment::Joined { .. } => panic!("expected a new shard, not a join"),
        }
        assert_eq!(registry.anchor_count(), 3);
    }

    #[test]
    fn test_loaded_registry_can_still_join_existing_anchors() {
        let existing_anchors = vec![ShardAnchor {
            shard_id: "periphery-0".to_string(),
            root_position: vec![0.92, 0.0, 0.0],
        }];
        let mut registry = ShardRegistry::load(existing_anchors, 1);
        let config = ShardingConfig::default();

        let assignment = registry.route(&[0.93, 0.01, 0.0], &config);

        match assignment {
            ShardAssignment::Joined { shard_id, .. } => assert_eq!(shard_id, "periphery-0"),
            ShardAssignment::NewShard { .. } => panic!("expected this point to join the loaded anchor"),
        }
        assert_eq!(registry.anchor_count(), 1, "joining shouldn't grow the loaded set");
    }

    #[test]
    fn test_accessors_expose_what_km_needs_to_persist() {
        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();
        registry.route(&[0.92, 0.0, 0.0], &config);

        assert_eq!(registry.anchors().len(), 1);
        assert_eq!(registry.next_id(), 1);
    }

    #[test]
    fn test_join_picks_nearest_of_multiple_anchors() {
        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();

        registry.route(&[0.92, 0.0, 0.0], &config); // periphery-0
        registry.route(&[-0.92, 0.0, 0.0], &config); // periphery-1

        // Closer to the second anchor than the first.
        let third = registry.route(&[-0.90, 0.02, 0.0], &config);

        match third {
            ShardAssignment::Joined { shard_id, .. } => {
                assert_eq!(shard_id, "periphery-1");
            }
            ShardAssignment::NewShard { .. } => panic!("expected a join, not a new shard"),
        }
        assert_eq!(registry.anchor_count(), 2, "should not have grown");
    }
}