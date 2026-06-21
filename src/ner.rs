use std::collections::HashSet;

use rake::{Rake, StopWords};
use stop_words::{get, LANGUAGE};

// ── Stopwords ─────────────────────────────────────────────────────────────────

/// Reuses the same stop-words crate the old tfidf.rs already depended on,
/// rather than hand-rolling a list — one source of truth for "what's noise"
/// instead of two lists that could quietly drift apart.
fn build_stopwords() -> StopWords {
    let set: HashSet<String> = get(LANGUAGE::English).into_iter().collect();
    StopWords::from(set)
}

// ── Capitalization heuristic ──────────────────────────────────────────────────

/// A "proper noun phrase" is a run of consecutive capitalized words. This
/// will mis-flag sentence-initial words sometimes (no easy way to tell
/// "The" starting a sentence from "The" in a real title without actual
/// sentence boundary detection) — same kind of noise spaCy's statistical
/// NER had anyway, and tfidf.rs's corpus-wide scoring is what's supposed
/// to wash out one-off noise like that, not this function.
fn capitalized_phrases(text: &str) -> HashSet<String> {
    let mut phrases = HashSet::new();
    let mut current: Vec<&str> = Vec::new();

    let is_cap_word = |w: &str| -> bool {
        w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) && w.chars().any(|c| c.is_alphabetic())
    };

    let flush = |current: &mut Vec<&str>, phrases: &mut HashSet<String>| {
        if !current.is_empty() {
            let phrase = current.join(" ").to_lowercase();
            if phrase.len() > 2 {
                phrases.insert(phrase);
            }
            current.clear();
        }
    };

    for raw_word in text.split_whitespace() {
        let trimmed = raw_word.trim_matches(|c: char| !c.is_alphanumeric());
        if trimmed.is_empty() {
            flush(&mut current, &mut phrases);
            continue;
        }
        if is_cap_word(trimmed) {
            current.push(trimmed);
        } else {
            flush(&mut current, &mut phrases);
        }
    }
    flush(&mut current, &mut phrases);

    phrases
}

// ── RAKE phrases ──────────────────────────────────────────────────────────────

fn rake_phrases(text: &str, rake: &Rake) -> HashSet<String> {
    // No score threshold here deliberately — RAKE already only returns
    // content phrases (it splits candidates apart at stopword/punctuation
    // boundaries, so stopword-only text never produces a candidate at
    // all). Real signal-vs-noise separation happens corpus-wide in
    // tfidf.rs, same division of labor extract_concepts_from_chunk /
    // build_tfidf had in the Python version — filtering here would risk
    // discarding a term before TF-IDF ever gets to evaluate it.
    rake.run(text)
        .into_iter()
        .map(|k| k.keyword.to_lowercase())
        .filter(|k| k.len() > 2)
        .collect()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Extracts candidate concept terms from a single chunk of text — the
/// direct replacement for spaCy's NER + noun_chunks in extract_concepts_
/// from_chunk. Combines RAKE candidate phrases with capitalized entity
/// phrases, deduped. No model load, no inference — pure text processing,
/// same complexity class as counting words.
pub fn extract_terms(text: &str, rake: &Rake) -> HashSet<String> {
    let mut terms = rake_phrases(text, rake);
    terms.extend(capitalized_phrases(text));
    terms
}

/// Builds the Rake instance once — reuse this across every chunk in a
/// corpus rather than rebuilding the stopword set per call.
pub fn build_extractor() -> Rake {
    Rake::new(build_stopwords())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extracts_proper_nouns() {
        let rake = build_extractor();
        let text = "SpaceX reported revenue growth. Elon Musk remains Chief Executive Officer.";
        let terms = extract_terms(text, &rake);

        assert!(terms.iter().any(|t| t.contains("spacex")));
        assert!(terms.iter().any(|t| t.contains("elon musk")));
    }

    #[test]
    fn test_stopword_only_text_yields_nothing() {
        let rake = build_extractor();
        let text = "the a an of to in on at";
        let terms = extract_terms(text, &rake);
        assert!(terms.is_empty());
    }

    #[test]
    fn test_deterministic_across_calls() {
        let rake = build_extractor();
        let text = "Jean Valjean spent nineteen years in the galleys at Toulon.";
        let a = extract_terms(text, &rake);
        let b = extract_terms(text, &rake);
        assert_eq!(a, b);
    }

    #[test]
    fn test_short_fragments_filtered() {
        let rake = build_extractor();
        let text = "ok an it is";
        let terms = extract_terms(text, &rake);
        assert!(terms.iter().all(|t| t.len() > 2));
    }
}