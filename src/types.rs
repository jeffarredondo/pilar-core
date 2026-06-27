use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// ── Chunk ─────────────────────────────────────────────────────────────────────

/// A single chunk of source text, with enough provenance to trace a
/// scored term back to where it came from. Produced by ingest.rs,
/// consumed by tfidf.rs — shared the same way Concept is, so it lives
/// here rather than being owned by either side.
pub struct Chunk {
    pub text: String,
    pub source_path: PathBuf,
    pub source_line: Option<usize>,
}

// ── Manifold Coordinate ──────────────────────────────────────────────────────

/// A concept's position on the product manifold M = H³ × S¹ × ℝⁿ, always in
/// LOCAL shard coordinates — already recentered via translate_to_origin if
/// this concept lives in a periphery shard. Which shard it belongs to is
/// implied by context (whichever Shard's map contains it), not stored here.
///
/// One variant per geometry, not one struct with a field per geometry —
/// the variant tag IS the geometry, so there's no separate "primary"
/// field that could ever drift out of sync with which fields are
/// actually populated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ManifoldCoord {
    /// Fixed at 3 dimensions deliberately — H³ was a named decision, not
    /// "however many happened to get passed in." A future move to H⁴+
    /// should be a conscious type change, not something that silently
    /// works because this was generic.
    Hyperbolic { position: [f64; 3] },
    /// Just the angle — (cos θ, sin θ) is one trig call away whenever
    /// it's actually needed, so storing the pair too would just be
    /// redundant state that could drift from the source angle.
    Spherical { theta: f64 },
    /// Still open how many flat dimensions a corpus actually needs —
    /// unlike H³, there was never a deliberate "it's exactly this many"
    /// moment, so Vec honestly reflects that this is still undecided.
    Flat { position: Vec<f64> },
}

// ── Geometry Confidence ──────────────────────────────────────────────────────

/// How clearly a concept's local neighborhood matched its assigned
/// geometry — the raw signal eigenvalue_signature and gromov_delta
/// produced on the way to a decision, kept instead of discarded.
///
/// This is provenance, not a second vote: distance always runs on
/// ManifoldCoord alone, never on these numbers. A concept that barely
/// landed hyperbolic and one that landed hyperbolic with zero ambiguity
/// get the identical ManifoldCoord variant — this is the only place
/// that distinction still exists. As more of a corpus gets ingested and
/// a concept's neighborhood fills in, these numbers are expected to
/// drift away from their classification thresholds in one direction or
/// the other — the signal sharpening as observation density increases.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GeometryConfidence {
    pub gromov_delta: f64,
    /// Spherical signal strength — ratio of 2nd to 1st normalized eigenvalue.
    pub eigenvalue_ratio: f64,
    /// Flat signal strength — how much the 1st eigenvalue dominates.
    pub first_dominance: f64,
    /// Hyperbolic signal strength — fraction of significantly negative eigenvalues.
    pub neg_eigenvalue_fraction: f64,
}

// ── Concept ───────────────────────────────────────────────────────────────────

/// A single concept placed on the manifold, plus everything needed to
/// trace it back to where it came from and re-derive its placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Deterministic extraction output (RAKE/capitalization) — same corpus,
    /// same text, same raw_term every run. The stable anchor for
    /// traceability, independent of whatever the naming step decides.
    pub raw_term: String,
    /// LLM-assigned name. Probabilistic, not deterministic, and not
    /// expected to be — one draw from a distribution over plausible
    /// names, not a fixed fact being computed correctly or incorrectly.
    pub label: String,
    /// Empty until the enrichment stage actually runs and writes a real
    /// value. No fallback, no placeholder borrowed from raw_term — an
    /// empty description is an honest "not enriched yet" signal, not
    /// something to paper over.
    pub description: String,
    pub coordinate: ManifoldCoord,
    pub confidence: GeometryConfidence,
    /// Stored once, reused for cross-geometry bridging — this is what
    /// fixes the old Python bug where match_to_disk re-embedded every
    /// disk concept on every single query instead of reusing this.
    #[serde(skip)]
    pub embedding: Vec<f64>,
    /// Normalized [0, 1] confidence signal — how reinforced is this concept.
    /// Source-agnostic deliberately: today it's always TF-IDF-derived, but
    /// nothing here should assume that. A future conversation-sourced
    /// concept (access count + time decay, say) would populate this the
    /// same way, with a completely different computation behind it. Distance
    /// and placement only ever see this number, never which producer made it.
    pub strength: f64,
    pub source_path: PathBuf,
    /// TOML has no native null -- a missing key, not a None value, is
    /// how "no line number" gets represented on disk. skip_serializing_if
    /// handles the write side; serde already treats a missing key as
    /// None for Option fields on the read side, no extra attribute needed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_line: Option<usize>,
}