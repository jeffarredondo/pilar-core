use std::collections::HashMap;
use std::path::PathBuf;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A single chunk of source text, with enough provenance to trace a
/// scored term back to where it came from.
pub struct Chunk {
    pub text: String,
    pub source_path: PathBuf,
    pub source_line: Option<usize>,
}

/// A term scored against the corpus it was extracted from.
#[derive(Debug, Clone)]
pub struct ScoredTerm {
    pub term: String,
    pub tfidf: f64,
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
pub fn compute(chunks: &[Chunk], terms_per_chunk: &[std::collections::HashSet<String>]) -> Vec<ScoredTerm> {
    debug_assert_eq!(chunks.len(), terms_per_chunk.len());

    let n = chunks.len();
    if n == 0 {
        return vec![];
    }

    // TF: total occurrences across all chunks
    let mut tf: HashMap<String, f64> = HashMap::new();
    // DF: how many distinct chunks contain this term
    let mut df: HashMap<String, usize> = HashMap::new();
    // First chunk a term was seen in, for source attribution
    let mut first_seen: HashMap<String, usize> = HashMap::new();

    for (i, terms) in terms_per_chunk.iter().enumerate() {
        for term in terms {
            *tf.entry(term.clone()).or_insert(0.0) += 1.0;
            *df.entry(term.clone()).or_insert(0) += 1;
            first_seen.entry(term.clone()).or_insert(i);
        }
    }

    // Smoothed only INSIDE the log (the +1 on n and df dodges a divide-
    // by-zero when df == n). Deliberately NOT smoothed outside the log —
    // an earlier version added +1.0 there too, which floors IDF so close
    // to 1.0 that raw term frequency overwhelms the document-frequency
    // penalty entirely, letting a term in every chunk outscore a term in
    // one chunk. A fully-ubiquitous term (df == n) correctly drives IDF
    // to ln(1) == 0, and a term that's MORE common than that can and
    // should go negative — that's the formula correctly saying "this is
    // stopword-like," not a bug to paper over.
    let mut results = Vec::with_capacity(tf.len());
    for (term, freq) in tf {
        let d = *df.get(&term).unwrap_or(&1) as f64;
        let idf = ((n as f64 + 1.0) / (d + 1.0)).ln();
        let score = freq * idf;

        let chunk_idx = *first_seen.get(&term).unwrap();
        let chunk = &chunks[chunk_idx];

        results.push(ScoredTerm {
            term,
            tfidf: score,
            source_path: chunk.source_path.clone(),
            source_line: chunk.source_line,
        });
    }

    results.sort_by(|a, b| b.tfidf.partial_cmp(&a.tfidf).unwrap());
    results
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

        let scores = compute(&chunks, &terms_per_chunk);
        let manifold = scores.iter().find(|s| s.term == "manifold").unwrap();
        let distance = scores.iter().find(|s| s.term == "distance").unwrap();

        // "manifold" appears in one chunk, "distance" in both —
        // manifold should score higher.
        assert!(manifold.tfidf > distance.tfidf);
    }

    #[test]
    fn test_empty_corpus() {
        let scores = compute(&[], &[]);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_source_attribution_points_to_first_occurrence() {
        let chunks = vec![chunk("first chunk"), chunk("second chunk")];
        let terms_per_chunk = vec![terms(&["recurring"]), terms(&["recurring"])];

        let scores = compute(&chunks, &terms_per_chunk);
        let recurring = scores.iter().find(|s| s.term == "recurring").unwrap();
        assert_eq!(recurring.source_path, chunks[0].source_path);
    }
}