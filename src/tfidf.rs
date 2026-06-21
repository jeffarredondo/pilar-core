use std::collections::HashMap;
use std::path::PathBuf;

use crate::types::Chunk;

// ── Config ────────────────────────────────────────────────────────────────────

pub struct TfidfConfig {
    /// Minimum total occurrences across the whole corpus for a term to
    /// be scored at all. Ported from Python's min_tf=10. A term
    /// extracted once or twice is almost certainly an extraction false
    /// positive, not a real recurring concept -- this filters that out
    /// before IDF is even computed, the same place Python's version did
    /// it. Unlike a top-N cap, this doesn't bound output size by a fixed
    /// count: it bounds it by "did this actually recur," so a bigger
    /// corpus with more genuinely-recurring concepts correctly produces
    /// more scored terms, not an artificially squashed-down number.
    pub min_occurrences: usize,
}

impl Default for TfidfConfig {
    fn default() -> Self {
        Self { min_occurrences: 10 }
    }
}

// ── Types ─────────────────────────────────────────────────────────────────────

/// A term scored against the corpus it was extracted from.
#[derive(Debug, Clone)]
pub struct ScoredTerm {
    pub term: String,
    pub tfidf: f64,
    /// Every chunk index this term appeared in, not just the first.
    /// enrich.rs needs this to rank a concept's own chunks by how much
    /// other high-strength signal co-occurs in them -- the Rust version
    /// of score_chunks_for_concept in the Python original, which needed
    /// the full chunk list, not just where a term was first seen.
    pub chunk_indices: Vec<usize>,
    /// First occurrence specifically, for source attribution --
    /// equivalent to chunk_indices[0], kept as its own field so callers
    /// don't need to know that to get a source location.
    pub source_path: PathBuf,
    pub source_line: Option<usize>,
}

// ── TF-IDF ────────────────────────────────────────────────────────────────────

/// Scores every term ner.rs extracted, using chunks as documents — IDF
/// measures how many chunks a term shows up in, not how many separate
/// source files. A term that appears in nearly every chunk scores near
/// zero (it's noise, not signal); a term concentrated in a few chunks
/// scores high.
///
/// Always called against the full corpus, every run — no persisted
/// running counts, no incremental update logic. At this corpus size
/// (low hundreds of chunks, no embeddings or LLM calls involved) this
/// is pure counting, cheap enough to just redo from scratch rather than
/// maintain state that could drift from whatever's actually on disk.
///
/// `terms_per_chunk` is ner.rs's extract_terms output, one HashSet per
/// chunk, same indices as `chunks`.
pub fn compute(chunks: &[Chunk], terms_per_chunk: &[std::collections::HashSet<String>], config: &TfidfConfig) -> Vec<ScoredTerm> {
    debug_assert_eq!(chunks.len(), terms_per_chunk.len());

    let n = chunks.len();
    if n == 0 {
        return vec![];
    }

    // TF: total occurrences across all chunks
    let mut tf: HashMap<String, f64> = HashMap::new();
    // DF: how many distinct chunks contain this term
    let mut df: HashMap<String, usize> = HashMap::new();
    // Every chunk index a term appeared in -- subsumes "first chunk
    // seen" (that's just chunk_indices[0]) while also giving enrich.rs
    // the full picture it needs.
    let mut chunk_indices: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, terms) in terms_per_chunk.iter().enumerate() {
        for term in terms {
            *tf.entry(term.clone()).or_insert(0.0) += 1.0;
            *df.entry(term.clone()).or_insert(0) += 1;
            chunk_indices.entry(term.clone()).or_default().push(i);
        }
    }

    // Smoothed only INSIDE the log (the +1 on n and df dodges a divide-
    // by-zero when df == n). Deliberately NOT smoothed outside the log —
    // an earlier version added +1.0 there too, which floors IDF so close
    // to 1.0 that raw term frequency overwhelms the document-frequency
    // penalty entirely, letting a term in every chunk outscore a term in
    // one chunk. Scores are bounded at exactly zero, never negative: df
    // can't exceed n by definition, so (n+1)/(df+1) >= 1 always, and a
    // fully-ubiquitous term (df == n) correctly bottoms out at ln(1) == 0
    // -- that's the formula correctly saying "this is stopword-like,"
    // not a bug to paper over.
    let mut results = Vec::with_capacity(tf.len());
    for (term, freq) in tf {
        // Below min_occurrences -- skip entirely, before IDF, before a
        // strength value ever exists for it. Matches Python's
        // `if tf[concept] < min_tf: continue` exactly.
        if freq < config.min_occurrences as f64 {
            continue;
        }

        let d = *df.get(&term).unwrap_or(&1) as f64;
        let idf = ((n as f64 + 1.0) / (d + 1.0)).ln();
        let score = freq * idf;

        let indices = chunk_indices.remove(&term).unwrap_or_default();
        let first_chunk_idx = indices[0];
        let chunk = &chunks[first_chunk_idx];

        results.push(ScoredTerm {
            term,
            tfidf: score,
            chunk_indices: indices,
            source_path: chunk.source_path.clone(),
            source_line: chunk.source_line,
        });
    }

    results.sort_by(|a, b| b.tfidf.partial_cmp(&a.tfidf).unwrap());
    results
}

// ── Strength normalization ───────────────────────────────────────────────────

/// Turns raw TF-IDF scores into the normalized [0, 1] `strength` signal
/// placement.rs actually consumes. This is tfidf.rs's own way of producing
/// that signal — a future conversation-sourced strength (access count +
/// time decay) would have its own equivalent function. Neither needs to
/// know the other exists; placement.rs only ever sees the normalized
/// output, never a raw tfidf score or which producer made it.
///
/// Scores from compute() are always >= 0 (a fully-ubiquitous term bottoms
/// out at exactly zero, never below — see compute()'s comment). The lower
/// clamp here is defensive rather than handling an observed case, in case
/// that ever changes; it's the upper clamp doing the real work, since
/// nothing otherwise guarantees the single highest score normalizes to
/// exactly 1.0 without floating-point noise.
pub fn normalize_to_strength(scores: &[ScoredTerm]) -> HashMap<String, f64> {
    let max_score = scores.iter().map(|s| s.tfidf).fold(f64::MIN, f64::max);

    if max_score <= 0.0 {
        // Every term in this batch is fully ubiquitous (score == 0) or
        // the batch is empty -- nothing to normalize against. Everything
        // gets zero strength rather than dividing by a non-positive max.
        return scores.iter().map(|s| (s.term.clone(), 0.0)).collect();
    }

    scores
        .iter()
        .map(|s| (s.term.clone(), (s.tfidf / max_score).clamp(0.0, 1.0)))
        .collect()
}

// ── Output (deferred) ─────────────────────────────────────────────────────────

// We may eventually want to dump scored terms to a pipe-delimited file
// for inspection before placement runs — easy to eyeball or load into
// anything without committing to a schema in code, and pipe-delimited
// sidesteps the comma-in-phrases problem ("Boca Chica, Texas" is a real
// RAKE candidate from this corpus). Not building it now: there's no
// real score output yet to decide the format against, and this is a
// write-only artifact nothing else reads back in, so there's no
// correctness risk to deferring it. Revisit once compute() has run
// against real data and we know what's actually worth inspecting.

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn chunk(text: &str) -> Chunk {
        Chunk {
            text: text.into(),
            source_path: PathBuf::from("test.md"),
            source_line: None,
        }
    }

    fn terms(words: &[&str]) -> HashSet<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    // Existing tests below use tiny hand-built fixtures (term
    // frequencies of 1-2) to exercise the scoring math itself, not the
    // occurrence floor -- this keeps min_occurrences out of their way.
    fn permissive_config() -> TfidfConfig {
        TfidfConfig { min_occurrences: 1 }
    }

    #[test]
    fn test_normalize_to_strength_basic() {
        let chunks = vec![
            chunk("hyperbolic geometry manifold distance"),
            chunk("hyperbolic space distance calculation"),
        ];
        let terms_per_chunk = vec![
            terms(&["hyperbolic", "geometry", "manifold", "distance"]),
            terms(&["hyperbolic", "space", "distance", "calculation"]),
        ];

        let scores = compute(&chunks, &terms_per_chunk, &permissive_config());
        let strengths = normalize_to_strength(&scores);

        // Highest-scored term should normalize to exactly 1.0
        let max = strengths.values().cloned().fold(f64::MIN, f64::max);
        assert!((max - 1.0).abs() < 1e-9);

        // Every value lands in [0, 1]
        assert!(strengths.values().all(|&s| (0.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_normalize_ubiquitous_term_bottoms_out_at_zero() {
        // "distance" appears in every chunk -- its raw score bottoms out
        // at exactly zero (see compute()'s comment: df can't exceed n,
        // so this is the floor, not a negative number). "rare" appears
        // in only one chunk and should score positively, so this test
        // actually exercises the per-term clamp path, not the all-zero
        // early return.
        let chunks = vec![chunk("a"), chunk("b"), chunk("c")];
        let terms_per_chunk = vec![
            terms(&["distance", "rare"]),
            terms(&["distance"]),
            terms(&["distance"]),
        ];

        let scores = compute(&chunks, &terms_per_chunk, &permissive_config());
        let strengths = normalize_to_strength(&scores);

        assert_eq!(strengths["distance"], 0.0);
        assert!(strengths["rare"] > 0.0);
    }

    #[test]
    fn test_normalize_empty_input() {
        let strengths = normalize_to_strength(&[]);
        assert!(strengths.is_empty());
    }

    #[test]
    fn test_chunk_indices_captures_every_occurrence() {
        let chunks = vec![chunk("a"), chunk("b"), chunk("c"), chunk("d")];
        let terms_per_chunk = vec![
            terms(&["recurring"]),
            terms(&["other"]),
            terms(&["recurring"]),
            terms(&["recurring"]),
        ];

        let scores = compute(&chunks, &terms_per_chunk, &permissive_config());
        let recurring = scores.iter().find(|s| s.term == "recurring").unwrap();

        // Appeared in chunks 0, 2, and 3 -- not just the first.
        assert_eq!(recurring.chunk_indices, vec![0, 2, 3]);
    }

    #[test]
    fn test_rare_term_scores_higher_than_common_term() {
        let chunks = vec![
            chunk("hyperbolic geometry manifold distance"),
            chunk("hyperbolic space distance calculation"),
        ];
        let terms_per_chunk = vec![
            terms(&["hyperbolic", "geometry", "manifold", "distance"]),
            terms(&["hyperbolic", "space", "distance", "calculation"]),
        ];

        let scores = compute(&chunks, &terms_per_chunk, &permissive_config());
        let manifold = scores.iter().find(|s| s.term == "manifold").unwrap();
        let distance = scores.iter().find(|s| s.term == "distance").unwrap();

        // "manifold" appears in one chunk, "distance" in both —
        // manifold should score higher.
        assert!(manifold.tfidf > distance.tfidf);
    }

    #[test]
    fn test_empty_corpus() {
        let scores = compute(&[], &[], &permissive_config());
        assert!(scores.is_empty());
    }

    #[test]
    fn test_min_occurrences_filters_terms_below_floor() {
        let chunks = vec![chunk("a"), chunk("b"), chunk("c"), chunk("d"), chunk("e")];
        let terms_per_chunk = vec![
            terms(&["frequent", "rare"]),
            terms(&["frequent"]),
            terms(&["frequent"]),
            terms(&["frequent"]),
            terms(&["frequent"]),
        ];
        // "frequent" occurs 5 times, "rare" occurs once.
        let config = TfidfConfig { min_occurrences: 3 };

        let scores = compute(&chunks, &terms_per_chunk, &config);

        assert!(scores.iter().any(|s| s.term == "frequent"), "should keep a term at or above the floor");
        assert!(scores.iter().all(|s| s.term != "rare"), "should drop a term below the floor entirely");
    }

    #[test]
    fn test_min_occurrences_at_exactly_the_floor_survives() {
        // Boundary check: a term occurring exactly min_occurrences times
        // should survive, not get caught by an off-by-one.
        let chunks = vec![chunk("a"), chunk("b")];
        let terms_per_chunk = vec![terms(&["twice"]), terms(&["twice"])];
        let config = TfidfConfig { min_occurrences: 2 };

        let scores = compute(&chunks, &terms_per_chunk, &config);
        assert!(scores.iter().any(|s| s.term == "twice"));
    }

    #[test]
    fn test_source_attribution_points_to_first_occurrence() {
        let chunks = vec![chunk("first chunk"), chunk("second chunk")];
        let terms_per_chunk = vec![terms(&["recurring"]), terms(&["recurring"])];

        let scores = compute(&chunks, &terms_per_chunk, &permissive_config());
        let recurring = scores.iter().find(|s| s.term == "recurring").unwrap();
        assert_eq!(recurring.source_path, chunks[0].source_path);
    }
}