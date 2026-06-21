use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::sharding::{ShardAnchor, ShardRegistry};
use crate::types::Concept;

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum KmError {
    Io(std::io::Error),
    Parse(toml::de::Error),
    Serialize(toml::ser::Error),
}

impl std::fmt::Display for KmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KmError::Io(e) => write!(f, "IO error: {e}"),
            KmError::Parse(e) => write!(f, "parse error: {e}"),
            KmError::Serialize(e) => write!(f, "serialize error: {e}"),
        }
    }
}

impl std::error::Error for KmError {}

impl From<std::io::Error> for KmError {
    fn from(e: std::io::Error) -> Self {
        KmError::Io(e)
    }
}
impl From<toml::de::Error> for KmError {
    fn from(e: toml::de::Error) -> Self {
        KmError::Parse(e)
    }
}
impl From<toml::ser::Error> for KmError {
    fn from(e: toml::ser::Error) -> Self {
        KmError::Serialize(e)
    }
}

// ── Shard ─────────────────────────────────────────────────────────────────────

/// One shard's worth of concepts on disk -- root or periphery, identical
/// shape either way, since periphery concepts already carry their own
/// recentered local coordinates by the time placement.rs hands them off.
///
/// Keyed by raw_term, not label. label is the LLM's probabilistic name
/// and can collide across concepts -- Python's own pipeline had to
/// disambiguate colliding names with a suffix. raw_term is deterministic
/// and guaranteed unique within a single placement run by construction:
/// tfidf.rs aggregates by unique term string, so every unique raw_term
/// produces exactly one ScoredTerm, which produces exactly one Concept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shard {
    pub shard_id: String,
    pub concepts: HashMap<String, Concept>,
}

pub fn write_shard(concepts: &[Concept], shard_id: &str, path: &Path) -> Result<(), KmError> {
    let map: HashMap<String, Concept> = concepts.iter().cloned().map(|c| (c.raw_term.clone(), c)).collect();
    let shard = Shard {
        shard_id: shard_id.to_string(),
        concepts: map,
    };
    let toml_str = toml::to_string_pretty(&shard)?;
    std::fs::write(path, toml_str)?;
    Ok(())
}

pub fn read_shard(path: &Path) -> Result<Shard, KmError> {
    let content = std::fs::read_to_string(path)?;
    let shard: Shard = toml::from_str(&content)?;
    Ok(shard)
}

// ── Registry persistence ──────────────────────────────────────────────────────

/// What gets written to disk to resume a ShardRegistry across runs.
/// next_id is persisted explicitly rather than re-derived from anchor
/// shard_ids on load -- see ShardRegistry::load's doc comment for why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrySnapshot {
    pub anchors: Vec<ShardAnchor>,
    pub next_id: usize,
}

pub fn write_registry(registry: &ShardRegistry, path: &Path) -> Result<(), KmError> {
    let snapshot = RegistrySnapshot {
        anchors: registry.anchors().to_vec(),
        next_id: registry.next_id(),
    };
    let toml_str = toml::to_string_pretty(&snapshot)?;
    std::fs::write(path, toml_str)?;
    Ok(())
}

pub fn read_registry(path: &Path) -> Result<ShardRegistry, KmError> {
    let content = std::fs::read_to_string(path)?;
    let snapshot: RegistrySnapshot = toml::from_str(&content)?;
    Ok(ShardRegistry::load(snapshot.anchors, snapshot.next_id))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sharding::{ShardAssignment, ShardingConfig};
    use crate::types::{GeometryConfidence, ManifoldCoord};
    use std::path::PathBuf;

    fn mock_concept(raw_term: &str) -> Concept {
        Concept {
            raw_term: raw_term.to_string(),
            label: format!("{raw_term}_label"),
            description: "a test description".to_string(),
            coordinate: ManifoldCoord::Hyperbolic {
                position: [0.1, 0.2, 0.3],
            },
            confidence: GeometryConfidence {
                gromov_delta: 0.5,
                eigenvalue_ratio: 0.3,
                first_dominance: 0.6,
                neg_eigenvalue_fraction: 0.4,
            },
            embedding: vec![0.1, 0.2, 0.3],
            strength: 0.87,
            source_path: PathBuf::from("test.md"),
            source_line: Some(42),
        }
    }

    // Unique-per-process temp paths -- cargo test runs files in parallel
    // by default, so a shared fixed filename would race.
    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("pilar_test_{label}_{}.km", std::process::id()))
    }

    #[test]
    fn test_shard_roundtrip() {
        let path = temp_path("shard_roundtrip");
        let concepts = vec![mock_concept("manifold"), mock_concept("geometry")];

        write_shard(&concepts, "root", &path).unwrap();
        let loaded = read_shard(&path).unwrap();

        assert_eq!(loaded.shard_id, "root");
        assert_eq!(loaded.concepts.len(), 2);
        let c = loaded.concepts.get("manifold").unwrap();
        assert_eq!(c.label, "manifold_label");
        assert!((c.strength - 0.87).abs() < 1e-10);
        assert_eq!(c.source_line, Some(42));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_shard_preserves_all_three_geometries() {
        let path = temp_path("geometries");

        let mut hyperbolic = mock_concept("h_concept");
        hyperbolic.coordinate = ManifoldCoord::Hyperbolic {
            position: [0.1, 0.2, 0.3],
        };
        let mut spherical = mock_concept("s_concept");
        spherical.coordinate = ManifoldCoord::Spherical { theta: 1.57 };
        let mut flat = mock_concept("f_concept");
        flat.coordinate = ManifoldCoord::Flat { position: vec![0.5] };

        write_shard(&[hyperbolic, spherical, flat], "root", &path).unwrap();
        let loaded = read_shard(&path).unwrap();

        match &loaded.concepts["h_concept"].coordinate {
            ManifoldCoord::Hyperbolic { position } => assert_eq!(*position, [0.1, 0.2, 0.3]),
            _ => panic!("expected hyperbolic"),
        }
        match &loaded.concepts["s_concept"].coordinate {
            ManifoldCoord::Spherical { theta } => assert!((theta - 1.57).abs() < 1e-9),
            _ => panic!("expected spherical"),
        }
        match &loaded.concepts["f_concept"].coordinate {
            ManifoldCoord::Flat { position } => assert_eq!(*position, vec![0.5]),
            _ => panic!("expected flat"),
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_concept_with_no_source_line_roundtrips() {
        // The actual reason for skip_serializing_if -- a None here used
        // to be the risky case with the toml crate specifically.
        let path = temp_path("no_source_line");
        let mut c = mock_concept("no_line");
        c.source_line = None;

        write_shard(&[c], "root", &path).unwrap();
        let loaded = read_shard(&path).unwrap();

        assert_eq!(loaded.concepts["no_line"].source_line, None);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_registry_roundtrip_preserves_routing_behavior() {
        let path = temp_path("registry");

        let mut registry = ShardRegistry::new();
        let config = ShardingConfig::default();
        registry.route(&[0.92, 0.0, 0.0], &config);
        registry.route(&[-0.92, 0.0, 0.0], &config);

        write_registry(&registry, &path).unwrap();
        let mut loaded = read_registry(&path).unwrap();

        assert_eq!(loaded.anchor_count(), 2);

        // The real test: a loaded registry has to keep routing correctly
        // -- continuing the id sequence, not colliding with what's
        // already on disk.
        let assignment = loaded.route(&[0.0, 0.92, 0.0], &config);
        match assignment {
            ShardAssignment::NewShard { shard_id, .. } => assert_eq!(shard_id, "periphery-2"),
            ShardAssignment::Joined { .. } => panic!("expected a new shard"),
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_read_nonexistent_file_returns_io_error_not_panic() {
        let result = read_shard(&PathBuf::from("/nonexistent/path/shard.km"));
        assert!(matches!(result, Err(KmError::Io(_))));
    }

    #[test]
    fn test_read_malformed_toml_returns_parse_error() {
        let path = temp_path("malformed");
        std::fs::write(&path, "this is not valid toml { [ }").unwrap();

        let result = read_shard(&path);
        assert!(matches!(result, Err(KmError::Parse(_))));

        std::fs::remove_file(path).ok();
    }
}