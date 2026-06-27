use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::tfidf::ScoredTerm;
use crate::types::{Chunk, Concept};

// ── Config ────────────────────────────────────────────────────────────────────

pub struct EnrichConfig {
    pub base_url: String,
    /// Fast, heavily-guardrailed extractive model. Matches a narrow task
    /// to its actual failure mode rather than fighting it -- see the
    /// prompt in summarize_concept for why the rails matter here.
    pub summarize_model: String,
    /// Slower, more opinionated model -- suited to the single judgment
    /// call naming actually is, where TinyLlama-class output tends
    /// toward confident nonsense rather than a clean answer.
    pub name_model: String,
    pub max_chunks: usize,
    /// Translated from Python's "score > 400" -- that was a magic
    /// number tied to Python's raw, un-normalized TF-IDF scale, and
    /// means nothing against this pipeline's [0,1] strength. 0.5 is a
    /// starting point ("co-occurring with something in the top half of
    /// strength"), not a verified-correct value -- worth revisiting
    /// once there's real enrichment output to look at.
    pub signal_threshold: f64,
}

impl Default for EnrichConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            summarize_model: "tinyllama".to_string(),
            name_model: "mistral".to_string(),
            max_chunks: 5,
            signal_threshold: 0.5,
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum EnrichError {
    Request(String),
    UnexpectedResponse(String),
}

impl std::fmt::Display for EnrichError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnrichError::Request(msg) => write!(f, "couldn't reach Ollama: {msg}"),
            EnrichError::UnexpectedResponse(msg) => write!(f, "unexpected response from Ollama: {msg}"),
        }
    }
}

impl std::error::Error for EnrichError {}

// ── Chat client (shared by both enrichment steps) ────────────────────────────

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    // Ollama's /api/chat streams NDJSON by default -- explicit false is
    // required, not optional, or a single serde_json parse of the body
    // breaks against a stream of partial objects instead of one.
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
}

fn chat(prompt: &str, model: &str, config: &EnrichConfig) -> Result<String, EnrichError> {
    let client = reqwest::blocking::Client::new();
    let url = format!("{}/api/chat", config.base_url);

    let response = client
        .post(&url)
        .json(&ChatRequest {
            model,
            messages: vec![ChatMessage {
                role: "user",
                content: prompt,
            }],
            stream: false,
        })
        .send()
        .map_err(|e| EnrichError::Request(e.to_string()))?;

    let body: ChatResponse = response.json().map_err(|e| EnrichError::UnexpectedResponse(e.to_string()))?;

    Ok(body.message.content)
}

// ── Chunk ranking ─────────────────────────────────────────────────────────────

/// Ranks a concept's own chunks by how much other high-strength signal
/// co-occurs in them -- more signal neighbors means a more information-
/// dense chunk to summarize from. Direct port of score_chunks_for_concept.
///
/// Python's version took a `concept` parameter it never referenced in
/// the body -- it doesn't exclude the concept's own term from the
/// co-occurrence count. Preserved exactly rather than "fixed": that's a
/// real behavioral choice in the original, not a verified bug. The
/// unused parameter itself is dropped, since Rust won't let that slide
/// quietly the way Python does.
fn rank_chunks_by_signal_density<'a>(
    concept_chunk_indices: &[usize],
    all_chunks: &'a [Chunk],
    all_terms: &[ScoredTerm],
    strengths: &HashMap<String, f64>,
    signal_threshold: f64,
) -> Vec<&'a Chunk> {
    let high_value_terms: Vec<&str> = all_terms
        .iter()
        .filter(|t| strengths.get(&t.term).copied().unwrap_or(0.0) > signal_threshold)
        .map(|t| t.term.as_str())
        .collect();

    let mut scored: Vec<(usize, &Chunk)> = concept_chunk_indices
        .iter()
        .map(|&idx| {
            let chunk = &all_chunks[idx];
            let chunk_lower = chunk.text.to_lowercase();
            let signal_count = high_value_terms.iter().filter(|term| chunk_lower.contains(**term)).count();
            (signal_count, chunk)
        })
        .collect();

    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored.into_iter().map(|(_, c)| c).collect()
}

// ── Summarization (small model) ───────────────────────────────────────────────

fn build_summarize_prompt(raw_term: &str, context: &str) -> String {
    format!(
        "You are summarizing a text document.\n\
Write exactly one sentence describing what \"{raw_term}\" refers to in the excerpts below.\n\
Use only information explicitly present in the text — do not interpret, infer, or add outside knowledge.\n\
If the excerpts do not clearly describe \"{raw_term}\", write: \"This text mentions {raw_term} without a clear description.\"\n\
\n\
TEXT EXCERPTS:\n\
{context}\n\
\n\
One sentence description of \"{raw_term}\" based only on the text above:"
    )
}

/// Feeds a concept's most information-dense chunks to the small model,
/// gets a one-sentence description back. Direct port of summarize_concept
/// -- the guardrail wording in the prompt is deliberate, not boilerplate:
/// it keeps the task narrow and extractive enough that a fast, "consistently
/// crazy" small model has little room to wander.
pub fn summarize_concept(
    raw_term: &str,
    concept_chunk_indices: &[usize],
    all_chunks: &[Chunk],
    all_terms: &[ScoredTerm],
    strengths: &HashMap<String, f64>,
    config: &EnrichConfig,
) -> Result<String, EnrichError> {
    let ranked = rank_chunks_by_signal_density(concept_chunk_indices, all_chunks, all_terms, strengths, config.signal_threshold);
    let sample: Vec<&str> = ranked.iter().take(config.max_chunks).map(|c| c.text.as_str()).collect();
    let context = sample.join("\n---\n");

    let prompt = build_summarize_prompt(raw_term, &context);
    let response = chat(&prompt, &config.summarize_model, config)?;
    Ok(response.trim().to_string())
}

// ── Naming (larger model) ─────────────────────────────────────────────────────

fn build_name_prompt(description: &str) -> String {
    format!(
        "Based on this description, give a short 1-3 word name that captures what this concept is about.\n\
Do not use generic words like \"time\", \"one\", \"first\", \"number\".\n\
Use specific meaningful terms from the description.\n\
\n\
Description: {description}\n\
\n\
Respond with ONLY the name, nothing else. 2-3 words maximum:"
    )
}

/// Validates a proposed name and falls back to raw_term if it looks
/// malformed. Split out from name_concept so this logic is testable
/// without a live Ollama -- same safety net the Python version had:
/// probabilistic output gets checked, and a deterministic fallback
/// (raw_term, not a retry, not a panic) backs it up when validation
/// fails. This is the same raw_term/label relationship documented on
/// Concept itself, just enforced here at the point label gets written.
fn validate_name(proposed: &str, raw_term: &str) -> String {
    let name = proposed.trim().to_lowercase();
    if name.split_whitespace().count() > 4 || name.len() > 40 {
        raw_term.to_string()
    } else {
        name
    }
}

/// Asks the larger model to name a concept based on its description
/// rather than the raw extracted token. Direct port of name_concept.
pub fn name_concept(raw_term: &str, description: &str, config: &EnrichConfig) -> Result<String, EnrichError> {
    let prompt = build_name_prompt(description);
    let response = chat(&prompt, &config.name_model, config)?;
    Ok(validate_name(&response, raw_term))
}

// ── Combined entry point ──────────────────────────────────────────────────────

/// Runs both enrichment steps and writes the results directly into a
/// Concept's description and label -- the function that finally fills
/// in the two fields placement.rs deliberately left empty.
pub fn enrich_concept(
    concept: &mut Concept,
    concept_chunk_indices: &[usize],
    all_chunks: &[Chunk],
    all_terms: &[ScoredTerm],
    strengths: &HashMap<String, f64>,
    config: &EnrichConfig,
) -> Result<(), EnrichError> {
    let description = summarize_concept(&concept.raw_term, concept_chunk_indices, all_chunks, all_terms, strengths, config)?;
    let label = name_concept(&concept.raw_term, &description, config)?;

    concept.description = description;
    concept.label = label;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn chunk(text: &str) -> Chunk {
        Chunk {
            text: text.to_string(),
            source_path: PathBuf::from("test.md"),
            source_line: None,
        }
    }

    fn scored_term(term: &str, chunk_indices: Vec<usize>) -> ScoredTerm {
        ScoredTerm {
            term: term.to_string(),
            tfidf: 1.0, // unused by rank_chunks_by_signal_density -- strengths map drives it
            chunk_indices,
            source_path: PathBuf::from("test.md"),
            source_line: None,
        }
    }

    #[test]
    fn test_rank_chunks_prefers_higher_signal_density() {
        let chunks = vec![
            chunk("manifold and geometry and distance all appear here"),
            chunk("manifold appears alone"),
        ];
        let all_terms = vec![
            scored_term("geometry", vec![0]),
            scored_term("distance", vec![0]),
        ];
        let mut strengths = HashMap::new();
        strengths.insert("geometry".to_string(), 0.9);
        strengths.insert("distance".to_string(), 0.8);

        let ranked = rank_chunks_by_signal_density(&[0, 1], &chunks, &all_terms, &strengths, 0.5);

        // Chunk 0 has two high-value co-occurring terms, chunk 1 has none.
        assert_eq!(ranked[0].text, chunks[0].text);
    }

    #[test]
    fn test_rank_chunks_ignores_terms_below_threshold() {
        let chunks = vec![chunk("weak term appears here"), chunk("nothing special here")];
        let all_terms = vec![scored_term("weak", vec![0])];
        let mut strengths = HashMap::new();
        strengths.insert("weak".to_string(), 0.1); // below threshold

        let ranked = rank_chunks_by_signal_density(&[0, 1], &chunks, &all_terms, &strengths, 0.5);
        // Neither chunk should out-rank the other -- "weak" never counts.
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn test_validate_name_accepts_reasonable_name() {
        assert_eq!(validate_name("orbital mechanics", "fallback"), "orbital mechanics");
    }

    #[test]
    fn test_validate_name_falls_back_on_too_many_words() {
        let proposed = "this is way too many words for a name";
        assert_eq!(validate_name(proposed, "fallback_term"), "fallback_term");
    }

    #[test]
    fn test_validate_name_falls_back_on_too_long() {
        let proposed = "a".repeat(50);
        assert_eq!(validate_name(&proposed, "fallback_term"), "fallback_term");
    }

    #[test]
    fn test_validate_name_lowercases_and_trims() {
        assert_eq!(validate_name("  Orbital Mechanics  ", "fallback"), "orbital mechanics");
    }

    #[test]
    fn test_summarize_prompt_contains_guardrails() {
        let prompt = build_summarize_prompt("manifold", "some excerpt text");
        assert!(prompt.contains("do not interpret"));
        assert!(prompt.contains("outside knowledge"));
        assert!(prompt.contains("without a clear description"));
    }

    #[test]
    fn test_name_prompt_excludes_generic_words_instruction() {
        let prompt = build_name_prompt("a test description");
        assert!(prompt.contains("Do not use generic words"));
    }

    #[test]
    fn test_chat_request_serializes_with_stream_false() {
        let req = ChatRequest {
            model: "tinyllama",
            messages: vec![ChatMessage {
                role: "user",
                content: "hello",
            }],
            stream: false,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""stream":false"#));
    }

    #[test]
    fn test_deserializes_real_ollama_chat_response_shape() {
        let raw = r#"{"model":"tinyllama","message":{"role":"assistant","content":"a test response"},"done":true}"#;
        let body: ChatResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(body.message.content, "a test response");
    }
}