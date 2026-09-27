use std::collections::HashSet;

use crate::ngram;
use crate::spacy_ner::{self, SpacyNer, SpacyNerConfig};

// ── Hybrid extractor ──────────────────────────────────────────────────────────
//
// Two extractors, two different failure modes:
//   - ngram.rs (sliding window) catches domain compounds spaCy has no label
//     for -- "starlink subscriber arpu", "philosopher's stone", "shares of
//     class". It also produces a lot of generic noise ("result", "segment",
//     "remained") that TF-IDF is left to filter out on its own.
//   - spacy_ner.rs (NER) catches precisely the opposite: exact, well-typed
//     entities ("q1 financial results (2026)"-style anchors) but nothing
//     outside its label set, and nothing that isn't a named entity at all.
//
// The handoff's fix is a union, not a replacement -- neither one subsumes
// the other, and testing showed AL's spaCy-only extraction beat Pilar's
// n-gram-only extraction specifically because the entity anchors were
// missing, not because n-grams were wrong to keep.

pub struct HybridExtractor {
    ngram: ngram::Extractor,
    /// None when spaCy/en_core_web_sm isn't available in this environment.
    /// Degrading to n-grams-only rather than making spaCy a hard
    /// dependency of every build -- see `build_hybrid_extractor` for the
    /// tradeoff this makes explicit at construction time.
    spacy: Option<SpacyNer>,
}

impl HybridExtractor {
    pub fn spacy_available(&self) -> bool {
        self.spacy.is_some()
    }
}

/// Builds both extractors. Returns an error only if the ngram side fails
/// to construct (it doesn't, today, but this keeps the signature honest
/// about future failure modes) -- a missing spaCy install is NOT fatal
/// here, it's logged to stderr once and the pipeline proceeds n-gram-only,
/// because losing the extraction method that's actually been validated
/// (n-grams) over one that's still being added (NER) would be the wrong
/// failure direction for a research pipeline mid-run.
pub fn build_hybrid_extractor() -> HybridExtractor {
    let spacy = match spacy_ner::build_extractor(&SpacyNerConfig::default()) {
        Ok(extractor) => Some(extractor),
        Err(e) => {
            eprintln!(
                "warning: spaCy NER unavailable ({e}) -- continuing with n-gram \
                 extraction only. See README for the spaCy setup this needs."
            );
            None
        }
    };

    HybridExtractor {
        ngram: ngram::build_extractor(),
        spacy,
    }
}

/// Same shape as ngram::extract_terms / spacy_ner::extract_entities --
/// pipeline.rs calls this exactly where it used to call ner::extract_terms.
/// A spaCy call failure on one chunk (as opposed to spaCy being entirely
/// unavailable) is treated the same way: logged, that chunk falls back to
/// n-grams alone rather than aborting the whole corpus over one bad chunk.
pub fn extract_terms(text: &str, extractor: &HybridExtractor) -> HashSet<String> {
    let mut terms = ngram::extract_terms(text, &extractor.ngram);

    if let Some(spacy) = &extractor.spacy {
        match spacy_ner::extract_entities(text, spacy) {
            Ok(entities) => terms.extend(entities),
            Err(e) => eprintln!("warning: spaCy NER failed on a chunk ({e}), continuing with n-grams for it"),
        }
    }

    terms
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_includes_ngram_output_regardless_of_spacy_availability() {
        let extractor = build_hybrid_extractor();
        let text = "Philosopher's stone was sought by alchemists for centuries.";
        let terms = extract_terms(text, &extractor);
        // "philosopher's stone" strips its apostrophe in tokenize() -- this
        // just confirms the n-gram side still contributes when spaCy is
        // unavailable in whatever environment runs this test.
        assert!(terms.iter().any(|t| t.contains("stone")), "got: {:?}", terms);
    }

    #[test]
    fn test_hybrid_adds_spacy_entities_when_available() {
        let extractor = build_hybrid_extractor();
        if !extractor.spacy_available() {
            eprintln!("skipping: spaCy not installed in this test environment");
            return;
        }
        let text = "SpaceX reported strong quarterly results. Elon Musk commented on Starlink.";
        let terms = extract_terms(text, &extractor);
        assert!(terms.iter().any(|t| t.contains("spacex")), "got: {:?}", terms);
        assert!(terms.iter().any(|t| t.contains("elon musk")), "got: {:?}", terms);
    }

    #[test]
    fn test_union_deduplicates_overlapping_terms() {
        let extractor = build_hybrid_extractor();
        let text = "SpaceX SpaceX SpaceX.";
        let terms = extract_terms(text, &extractor);
        // Whatever the union produces, "spacex" (from n-grams at minimum)
        // appears exactly once as a set member -- the point of using a
        // HashSet at all rather than concatenating two Vecs.
        assert_eq!(terms.iter().filter(|t| t.as_str() == "spacex").count(), 1);
    }
}
