use std::collections::HashSet;

use stop_words::{get, LANGUAGE};

// ── Stopwords ─────────────────────────────────────────────────────────────────

/// Same stop-words crate tfidf.rs already uses — one source of truth for
/// "what's noise" rather than two lists that could quietly drift apart.
fn build_stopword_set() -> HashSet<String> {
    get(LANGUAGE::English).into_iter().collect()
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Controls n-gram extraction. `max_n` is deliberately left as a runtime
/// parameter rather than a compile-time constant — the right value isn't
/// known yet and needs to be determined empirically by running real corpora
/// through different sizes and inspecting output. Start with 3 and adjust.
pub struct NerConfig {
    /// Maximum n-gram length to emit. Unigrams (n=1) through max_n are
    /// all candidates — corpus-wide TF-IDF and min_occurrences do the
    /// actual curation, not this layer.
    pub max_n: usize,
}

impl Default for NerConfig {
    fn default() -> Self {
        Self { max_n: 3 }
    }
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

/// Strips punctuation from both ends of a word, lowercases, drops empties.
/// Inline rather than a crate — the only thing needed is "alphanumeric
/// characters of a token, lowercase," not a full tokenization pipeline.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|raw| {
            let t = raw.trim_matches(|c: char| !c.is_alphanumeric());
            if t.is_empty() { None } else { Some(t.to_lowercase()) }
        })
        .collect()
}

// ── N-gram extraction ─────────────────────────────────────────────────────────

/// Slides windows of size 1..=max_n over the token stream and keeps every
/// window whose first and last token are both non-stopwords. Internal
/// stopwords are fine — "bank of america" passes (edges are content words),
/// "of the bank" fails (starts on a stopword). Single-token windows are
/// included so genuine anchor concepts like "witchcraft" survive on their
/// own rather than only ever appearing folded into a bigram.
///
/// Overlap between n-grams of different lengths ("spacex", "spacex rocket",
/// "rocket") is intentional and not resolved here. Consolidating subsuming
/// concepts is a future vacuum/maintenance step over the manifold, not
/// ingestion's job — same reasoning as the Gutenberg-ranking-#1 call: don't
/// editorialize at extraction time about what's "really" one concept vs.
/// several, let density and geometry decide later.
fn ngram_terms(tokens: &[String], stopwords: &HashSet<String>, max_n: usize) -> HashSet<String> {
    let mut terms = HashSet::new();

    for n in 1..=max_n {
        for window in tokens.windows(n) {
            let first = &window[0];
            let last = &window[window.len() - 1];

            // Both edges must be content words — internal stopwords are fine.
            if stopwords.contains(first) || stopwords.contains(last) {
                continue;
            }

            let phrase = window.join(" ");

            // Mirror the old length floor — single characters and two-char
            // fragments are noise by any measure.
            if phrase.len() > 2 {
                terms.insert(phrase);
            }
        }
    }

    terms
}

// ── Extractor ─────────────────────────────────────────────────────────────────

/// Holds the stopword set so it's built once and reused across every chunk
/// in a corpus rather than reconstructed per call.
pub struct Extractor {
    stopwords: HashSet<String>,
    config: NerConfig,
}

impl Extractor {
    pub fn new(config: NerConfig) -> Self {
        Self {
            stopwords: build_stopword_set(),
            config,
        }
    }
}

impl Default for Extractor {
    fn default() -> Self {
        Self::new(NerConfig::default())
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Extracts candidate concept terms from a single chunk of text. Replaces
/// the old RAKE + capitalized_phrases combination — both had the same
/// underlying fragmentation bug (RAKE splits at stopwords, capitalized_phrases
/// splits at non-capitalized words, both fragment "Bank of America" at "of").
/// Sliding-window n-grams generalize RAKE's idea without that constraint:
/// edges must be content words, internal stopwords are allowed.
///
/// Output type is unchanged — still `HashSet<String>` — so tfidf.rs,
/// embed.rs, placement.rs, and enrich.rs need zero changes.
pub fn extract_terms(text: &str, extractor: &Extractor) -> HashSet<String> {
    let tokens = tokenize(text);
    ngram_terms(&tokens, &extractor.stopwords, extractor.config.max_n)
}

/// Builds the extractor once. Pass the result into every `extract_terms`
/// call for a corpus — same usage pattern as the old `build_extractor`/`Rake`.
pub fn build_extractor() -> Extractor {
    Extractor::default()
}

/// Builds the extractor with explicit config — use this when you want to
/// compare max_n values across runs rather than taking the default.
pub fn build_extractor_with_config(config: NerConfig) -> Extractor {
    Extractor::new(config)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn stopwords() -> HashSet<String> {
        build_stopword_set()
    }

    // ── ngram_terms unit tests ────────────────────────────────────────────────

    #[test]
    fn test_unigram_content_word_included() {
        let sw = stopwords();
        let tokens = vec!["witchcraft".to_string()];
        let terms = ngram_terms(&tokens, &sw, 3);
        assert!(terms.contains("witchcraft"));
    }

    #[test]
    fn test_unigram_stopword_excluded() {
        let sw = stopwords();
        let tokens = vec!["the".to_string()];
        let terms = ngram_terms(&tokens, &sw, 3);
        assert!(terms.is_empty());
    }

    #[test]
    fn test_bigram_internal_stopword_allowed() {
        // "bank of america" — "of" is a stopword but edges are content words.
        let sw = stopwords();
        let tokens = vec!["bank".to_string(), "of".to_string(), "america".to_string()];
        let terms = ngram_terms(&tokens, &sw, 3);
        assert!(terms.contains("bank of america"), "got: {:?}", terms);
    }

    #[test]
    fn test_bigram_leading_stopword_excluded() {
        // "of the bank" — starts on a stopword, should be dropped.
        let sw = stopwords();
        let tokens = vec!["of".to_string(), "the".to_string(), "bank".to_string()];
        let terms = ngram_terms(&tokens, &sw, 3);
        assert!(!terms.contains("of the bank"), "got: {:?}", terms);
    }

    #[test]
    fn test_bigram_trailing_stopword_excluded() {
        let sw = stopwords();
        let tokens = vec!["bank".to_string(), "of".to_string()];
        let terms = ngram_terms(&tokens, &sw, 2);
        assert!(!terms.contains("bank of"), "got: {:?}", terms);
    }

    #[test]
    fn test_ngrams_up_to_max_n_only() {
        let sw = stopwords();
        // Four content words — with max_n=2 we should get unigrams and bigrams
        // but no trigrams or 4-grams.
        let tokens = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string(), "delta".to_string()];
        let terms = ngram_terms(&tokens, &sw, 2);
        assert!(terms.contains("alpha beta"));
        assert!(!terms.contains("alpha beta gamma"));
    }

    #[test]
    fn test_length_floor_applied() {
        let sw = stopwords();
        // Two-character token shouldn't survive even if it's not a stopword.
        let tokens = vec!["ok".to_string()];
        let terms = ngram_terms(&tokens, &sw, 1);
        assert!(terms.is_empty());
    }

    // ── extract_terms integration tests ──────────────────────────────────────

    #[test]
    fn test_extracts_proper_nouns() {
        let extractor = build_extractor();
        let text = "SpaceX reported revenue growth. Elon Musk remains Chief Executive Officer.";
        let terms = extract_terms(text, &extractor);

        assert!(terms.iter().any(|t| t.contains("spacex")), "got: {:?}", terms);
        assert!(terms.iter().any(|t| t.contains("elon musk")), "got: {:?}", terms);
    }

    #[test]
    fn test_stopword_only_text_yields_nothing() {
        let extractor = build_extractor();
        let text = "the a an of to in on at";
        let terms = extract_terms(text, &extractor);
        assert!(terms.is_empty(), "got: {:?}", terms);
    }

    #[test]
    fn test_deterministic_across_calls() {
        let extractor = build_extractor();
        let text = "Jean Valjean spent nineteen years in the galleys at Toulon.";
        let a = extract_terms(text, &extractor);
        let b = extract_terms(text, &extractor);
        assert_eq!(a, b);
    }

    #[test]
    fn test_short_fragments_filtered() {
        let extractor = build_extractor();
        let text = "ok an it is";
        let terms = extract_terms(text, &extractor);
        assert!(terms.iter().all(|t| t.len() > 2));
    }

    #[test]
    fn test_internal_stopword_phrase_survives() {
        // The old RAKE path would fragment "bank of america" at "of".
        // N-gram extraction should keep it intact.
        let extractor = build_extractor();
        let text = "Bank of America reported earnings.";
        let terms = extract_terms(text, &extractor);
        assert!(terms.iter().any(|t| t == "bank of america"), "got: {:?}", terms);
    }

    #[test]
    fn test_max_n_config_respected() {
        let extractor = build_extractor_with_config(NerConfig { max_n: 1 });
        let text = "orbital mechanics governs trajectories.";
        let terms = extract_terms(text, &extractor);
        // With max_n=1, no multi-word phrases should appear.
        assert!(terms.iter().all(|t| !t.contains(' ')), "got: {:?}", terms);
    }

    #[test]
    fn test_max_n_2_produces_bigrams() {
        let extractor = build_extractor_with_config(NerConfig { max_n: 2 });
        let text = "orbital mechanics governs trajectories.";
        let terms = extract_terms(text, &extractor);
        assert!(terms.iter().any(|t| t.contains(' ')), "got: {:?}", terms);
    }
}