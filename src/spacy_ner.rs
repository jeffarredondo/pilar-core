use std::collections::HashSet;

use pyo3::prelude::*;

// ── Config ────────────────────────────────────────────────────────────────────

/// Which spaCy entity labels are worth pulling into the manifold. This is
/// deliberately a subset of spaCy's full label set (which also includes
/// things like CARDINAL, ORDINAL, LANGUAGE, LAW, NORP -- too generic or
/// too noisy for concept extraction). Matches the handoff's shortlist:
/// organizations, people, places, products, events, dates, money.
const DEFAULT_LABELS: &[&str] = &["ORG", "PERSON", "GPE", "PRODUCT", "EVENT", "DATE", "MONEY"];

pub struct SpacyNerConfig {
    /// spaCy model name to load. en_core_web_sm is the one the handoff
    /// calls out -- small, fast, no GPU needed, good enough for the
    /// entity types above.
    pub model: String,
    pub labels: HashSet<String>,
}

impl Default for SpacyNerConfig {
    fn default() -> Self {
        Self {
            model: "en_core_web_sm".to_string(),
            labels: DEFAULT_LABELS.iter().map(|s| s.to_string()).collect(),
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum SpacyNerError {
    /// spaCy (or the model) isn't importable in whatever Python `python3`
    /// resolved to at build time. Distinct from a runtime call failure so
    /// callers can decide whether "no spaCy available" should be fatal or
    /// just a reason to fall back to n-grams alone.
    ModelLoad(String),
    Call(String),
}

impl std::fmt::Display for SpacyNerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpacyNerError::ModelLoad(msg) => write!(
                f,
                "couldn't load spaCy model: {msg} -- is spaCy installed for this \
                 interpreter? (pip install spacy && python -m spacy download en_core_web_sm)"
            ),
            SpacyNerError::Call(msg) => write!(f, "spaCy NER call failed: {msg}"),
        }
    }
}

impl std::error::Error for SpacyNerError {}

// ── Cleaning ──────────────────────────────────────────────────────────────────
//
// Pure, spaCy-independent so these can be tested directly rather than
// relying on the model to reproduce a specific quirk (e.g. splitting an
// entity across a line break needs real document-length context around it
// -- a tiny isolated test sentence won't trigger it even with a literal
// newline inserted, verified against en_core_web_sm directly).

/// Collapses internal whitespace (not just trims the ends) and lowercases.
/// Source text pulled from PDF-to-text conversion sometimes has a literal
/// line break mid-entity ("elon\nmusk"), which spaCy's `ent.text` preserves
/// verbatim. Left alone, that term would never match ngram.rs's "elon musk"
/// (which normalizes via tokenize/join), so the two extractors would
/// produce two different strings for the same real-world entity instead of
/// the hybrid set converging on one.
fn normalize_entity_text(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// True if a MONEY entity looks like financial-statement-table bleed
/// rather than a real dollar figure. Tables in an S-1 render as runs of
/// figures with no real separators once PDF-to-text flattens them
/// ("$ 1,047 $ 727 $ 6,785 $ 5,776 $"), and spaCy's MONEY tagger glues the
/// whole run into one entity. A genuine dollar amount has exactly one "$"
/// in it -- more than that is table bleed, not a concept.
fn is_table_bleed_money(label: &str, cleaned: &str) -> bool {
    label == "MONEY" && cleaned.matches('$').count() > 1
}

/// SEC filings write a loss as "$(1,943) million" -- parens mean negative.
/// spaCy's MONEY span for this pattern starts AFTER the "$(" but still
/// includes the closing ")", so `ent.text` comes out as "1,943) million":
/// no dollar sign, no minus sign, nothing distinguishing it from a
/// positive figure. Confirmed directly against en_core_web_sm, not
/// assumed. Left alone, this is worse than losing the figure entirely --
/// a $(1,943) million loss and a $1,943 million gain become the exact
/// same term, so a query asking for a loss can retrieve this concept and
/// have no way to know it was actually negative.
///
/// Detected by an unbalanced trailing ")" with no matching "(" in the
/// entity's own text -- specific to how spaCy truncates this pattern, not
/// a general parenthesis-balance checker. Repairs it into an explicit
/// negative ("-$1,943 million") rather than trying to recover the
/// original span's exact characters.
fn repair_parenthetical_negative_money(label: &str, text: &str) -> String {
    if label != "MONEY" {
        return text.to_string();
    }

    let opens = text.matches('(').count();
    let closes = text.matches(')').count();
    if closes <= opens {
        return text.to_string();
    }

    let without_paren = text.replacen(')', "", 1);
    let trimmed = without_paren.trim_start();
    if trimmed.starts_with('$') {
        format!("-{trimmed}")
    } else {
        format!("-${trimmed}")
    }
}

// ── Extractor ─────────────────────────────────────────────────────────────────

/// Holds the loaded spaCy pipeline so `nlp` is built exactly once and
/// reused across every chunk in a corpus. Reloading en_core_web_sm per
/// chunk was flagged in the handoff as the trap to avoid -- model load is
/// the expensive part (tens to hundreds of ms), calling an already-loaded
/// pipeline on one chunk of text is fast.
pub struct SpacyNer {
    nlp: Py<PyAny>,
    labels: HashSet<String>,
}

/// If this process is running inside an activated virtualenv (`VIRTUAL_ENV`
/// set -- true whenever `source .venv/bin/activate` was used), make sure its
/// `site-packages` is actually on the embedded interpreter's `sys.path`.
///
/// PyO3's `auto-initialize` links against a Python found at *build* time,
/// but at *runtime* the embedded interpreter discovers its own sys.path
/// starting from the Rust binary's location, not from the venv's `python3`
/// launcher script -- so it can end up not seeing the venv's site-packages
/// at all even when the binary was correctly linked against the venv's
/// interpreter. This shows up as a spurious "No module named 'spacy'" even
/// though `pip show spacy` finds it fine from an activated shell. Observed
/// concretely on macOS with a framework-build Python; harmless no-op
/// everywhere else (outside a venv, or where the interpreter already
/// resolves this correctly on its own).
fn ensure_venv_site_packages_on_path(py: Python<'_>) -> PyResult<()> {
    let Ok(venv) = std::env::var("VIRTUAL_ENV") else {
        return Ok(());
    };

    let sys = py.import("sys")?;
    let version_info = sys.getattr("version_info")?;
    let major: u32 = version_info.get_item(0)?.extract()?;
    let minor: u32 = version_info.get_item(1)?.extract()?;

    // POSIX venv layout. Windows venvs use Lib\site-packages instead, but
    // the pipeline (Ollama, shell scripts) already assumes a POSIX
    // environment elsewhere, so this doesn't try to handle both.
    let site_packages = format!("{venv}/lib/python{major}.{minor}/site-packages");
    if !std::path::Path::new(&site_packages).is_dir() {
        return Ok(());
    }

    let sys_path = sys.getattr("path")?;
    let already_present: bool = sys_path.call_method1("__contains__", (&site_packages,))?.extract()?;
    if !already_present {
        sys_path.call_method1("insert", (0, &site_packages))?;
    }
    Ok(())
}

impl SpacyNer {
    /// Imports spaCy and loads the model once. `auto-initialize` means the
    /// pyo3 feature already brings up the interpreter on first use, so
    /// this just needs to do the Python-side `spacy.load(...)`.
    pub fn new(config: &SpacyNerConfig) -> Result<Self, SpacyNerError> {
        let nlp = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
            ensure_venv_site_packages_on_path(py)?;
            let spacy = py.import("spacy")?;
            let nlp = spacy.call_method1("load", (config.model.as_str(),))?;
            Ok(nlp.unbind())
        })
        .map_err(|e| SpacyNerError::ModelLoad(e.to_string()))?;

        Ok(Self {
            nlp,
            labels: config.labels.clone(),
        })
    }

    /// Runs the loaded pipeline over one chunk of text and returns every
    /// entity whose label is in the configured whitelist, lowercased to
    /// match ngram.rs's output convention (tfidf.rs and downstream treat
    /// terms as case-insensitive keys throughout).
    ///
    /// GIL is acquired per call via `Python::with_gil`, not per corpus --
    /// that's the "fine" half of the handoff's caution: reacquiring the
    /// GIL per chunk costs nothing next to actually running the model.
    pub fn extract(&self, text: &str) -> Result<HashSet<String>, SpacyNerError> {
        Python::with_gil(|py| -> PyResult<HashSet<String>> {
            let doc = self.nlp.bind(py).call1((text,))?;
            let ents = doc.getattr("ents")?;

            // spaCy's `doc.ents` is a tuple, not a list -- `try_iter`
            // works against any Python iterable rather than assuming a
            // concrete container type, so this doesn't care which one
            // spaCy happens to hand back.
            let mut out = HashSet::new();
            for ent in ents.try_iter()? {
                let ent = ent?;
                let label: String = ent.getattr("label_")?.extract()?;
                if !self.labels.contains(&label) {
                    continue;
                }
                let text: String = ent.getattr("text")?.extract()?;
                let repaired = repair_parenthetical_negative_money(&label, &text);
                let cleaned = normalize_entity_text(&repaired);
                if cleaned.is_empty() || is_table_bleed_money(&label, &cleaned) {
                    continue;
                }

                out.insert(cleaned);
            }
            Ok(out)
        })
        .map_err(|e| SpacyNerError::Call(e.to_string()))
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Builds the extractor once. Mirrors ngram.rs's `build_extractor` naming
/// so pipeline.rs's call sites read the same way for both extractors.
pub fn build_extractor(config: &SpacyNerConfig) -> Result<SpacyNer, SpacyNerError> {
    SpacyNer::new(config)
}

/// Extracts named entities from one chunk of text using an already-built
/// extractor. Free function (rather than only a method) to mirror
/// ngram.rs's `extract_terms(text, &extractor)` call shape exactly --
/// extraction.rs's hybrid union reads as two nearly-identical lines.
pub fn extract_entities(text: &str, extractor: &SpacyNer) -> Result<HashSet<String>, SpacyNerError> {
    extractor.extract(text)
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// These hit a real Python + spaCy + en_core_web_sm, unlike the rest of the
// suite -- they're integration tests of the FFI boundary itself, not pure
// unit tests. They'll fail (not silently skip) in an environment without
// spaCy installed, which is the intended signal: if this module can't
// actually load the model, that should show up loudly in CI, not get
// papered over.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loads_model_and_extracts_org_and_person() {
        let extractor = build_extractor(&SpacyNerConfig::default()).expect("spaCy model should load");
        let text = "SpaceX reported revenue growth. Elon Musk remains Chief Executive Officer.";
        let ents = extract_entities(text, &extractor).expect("extraction should succeed");

        assert!(ents.iter().any(|e| e.contains("spacex")), "got: {:?}", ents);
        assert!(ents.iter().any(|e| e.contains("elon musk")), "got: {:?}", ents);
    }

    #[test]
    fn test_filters_to_configured_labels_only() {
        // "nineteen years" is a DATE-adjacent quantity spaCy tags as
        // CARDINAL/DATE depending on phrasing; the real assertion here is
        // that whatever comes back is drawn only from the whitelist, not
        // that a specific label fires on this exact sentence.
        let config = SpacyNerConfig {
            labels: ["PERSON".to_string()].into_iter().collect(),
            ..SpacyNerConfig::default()
        };
        let extractor = build_extractor(&config).expect("spaCy model should load");
        let text = "Jean Valjean spent nineteen years in the galleys at Toulon.";
        let ents = extract_entities(text, &extractor).expect("extraction should succeed");

        assert!(ents.iter().any(|e| e.contains("valjean")), "got: {:?}", ents);
        // Toulon is a GPE, which was excluded by the label whitelist above.
        assert!(!ents.iter().any(|e| e == "toulon"), "got: {:?}", ents);
    }

    #[test]
    fn test_reused_extractor_is_deterministic() {
        let extractor = build_extractor(&SpacyNerConfig::default()).expect("spaCy model should load");
        let text = "The Brandenburg Concertos were composed by Johann Sebastian Bach.";
        let a = extract_entities(text, &extractor).unwrap();
        let b = extract_entities(text, &extractor).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_empty_text_yields_no_entities() {
        let extractor = build_extractor(&SpacyNerConfig::default()).expect("spaCy model should load");
        let ents = extract_entities("", &extractor).unwrap();
        assert!(ents.is_empty());
    }

    // ── normalize_entity_text / is_table_bleed_money (pure, no spaCy) ────────

    #[test]
    fn test_normalize_collapses_internal_line_break() {
        // The actual artifact observed against a real S-1: a name split
        // across a line break by PDF-to-text conversion, which ent.text
        // would otherwise preserve verbatim.
        assert_eq!(normalize_entity_text("Elon \nMusk"), "elon musk");
    }

    #[test]
    fn test_normalize_collapses_multiple_internal_spaces() {
        assert_eq!(normalize_entity_text("Goldman   Sachs  &  Co."), "goldman sachs & co.");
    }

    #[test]
    fn test_normalize_trims_and_lowercases() {
        assert_eq!(normalize_entity_text("  SpaceX  "), "spacex");
    }

    #[test]
    fn test_single_dollar_money_not_flagged_as_bleed() {
        // The exact case the handoff called out -- must survive.
        assert!(!is_table_bleed_money("MONEY", "$4,694 million"));
    }

    #[test]
    fn test_multi_dollar_money_flagged_as_bleed() {
        // What a flattened financial-statement table produces -- several
        // cells glued into one MONEY entity, not a real concept.
        assert!(is_table_bleed_money("MONEY", "$ 1,047 $ 727 $ 6,785 $ 5,776 $"));
    }

    #[test]
    fn test_multi_dollar_non_money_label_not_flagged() {
        // The bleed check is MONEY-specific -- an ORG or other label
        // containing a stray "$" (unlikely, but not this function's job
        // to guess about) shouldn't be caught by a rule aimed at tables.
        assert!(!is_table_bleed_money("ORG", "$ 1,047 $ 727 $"));
    }

    // ── repair_parenthetical_negative_money (pure, no spaCy) ─────────────────

    #[test]
    fn test_repairs_truncated_sec_negative() {
        // The exact real-world case: spaCy drops the leading "$(" from
        // "$(1,943) million" and hands back "1,943) million".
        assert_eq!(
            repair_parenthetical_negative_money("MONEY", "1,943) million"),
            "-$1,943 million"
        );
    }

    #[test]
    fn test_balanced_parens_left_alone() {
        // A term that happens to contain a complete, balanced parenthetical
        // isn't this bug -- nothing to repair.
        let balanced = "spacex (delaware)";
        assert_eq!(repair_parenthetical_negative_money("MONEY", balanced), balanced);
    }

    #[test]
    fn test_non_money_label_unbalanced_paren_left_alone() {
        // This repair is specific to the SEC-negative-MONEY pattern --
        // an unbalanced ")" on some other label isn't assumed to mean
        // the same thing.
        let text = "some org) fragment";
        assert_eq!(repair_parenthetical_negative_money("ORG", text), text);
    }

    #[test]
    fn test_repair_does_not_double_up_dollar_sign() {
        // Defensive: if the truncated text somehow already carries a "$",
        // don't produce "-$$1,943 million".
        assert_eq!(
            repair_parenthetical_negative_money("MONEY", "$1,943) million"),
            "-$1,943 million"
        );
    }

    // ── Integration: confirms the real SEC-filing sentence repairs end to
    // end, and that the repaired value survives the table-bleed filter
    // (exactly one "$", so it shouldn't be mistaken for table garbage) ──────

    #[test]
    fn test_sec_style_loss_figure_repaired_end_to_end() {
        let config = SpacyNerConfig {
            labels: ["MONEY".to_string()].into_iter().collect(),
            ..SpacyNerConfig::default()
        };
        let extractor = build_extractor(&config).expect("spaCy model should load");
        let text = "For the three months ended March 31, 2026, we generated revenue on a \
                     consolidated basis of $4,694 million, loss from operations of $(1,943) \
                     million and Adjusted EBITDA of $1,127 million.";
        let ents = extract_entities(text, &extractor).expect("extraction should succeed");

        assert!(
            ents.iter().any(|e| e.contains("-$1,943 million") || e.contains("-1,943 million")),
            "expected a negative loss figure, got: {:?}",
            ents
        );
        // Revenue and EBITDA are positive and must not have picked up a
        // stray minus sign from this fix.
        assert!(ents.iter().any(|e| e == "$4,694 million"), "got: {:?}", ents);
        assert!(!ents.iter().any(|e| e.contains('(') || e.contains(')')), "got: {:?}", ents);
    }

    // ── Integration: still confirms the real FFI path produces a clean,
    // single-$ MONEY entity end to end ────────────────────────────────────

    #[test]
    fn test_single_dollar_money_entity_survives_end_to_end() {
        let config = SpacyNerConfig {
            labels: ["MONEY".to_string()].into_iter().collect(),
            ..SpacyNerConfig::default()
        };
        let extractor = build_extractor(&config).expect("spaCy model should load");
        let text = "SpaceX reported revenue of $4,694 million for the quarter.";
        let ents = extract_entities(text, &extractor).expect("extraction should succeed");

        assert!(ents.iter().any(|e| e.contains("4,694")), "got: {:?}", ents);
    }
}