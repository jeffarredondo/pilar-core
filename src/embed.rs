use serde::{Deserialize, Serialize};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Deserialize)] 
pub struct EmbedConfig {
    pub base_url: String,
    pub model: String,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "nomic-embed-text".to_string(),
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum EmbedError {
    /// Couldn't reach Ollama at all -- not running, wrong port, network down.
    Request(String),
    /// Got a response, but it wasn't shaped the way Ollama's docs say it
    /// should be -- wrong model name (Ollama returns an error body, not
    /// an embeddings array), or an API shape change we haven't caught up to.
    UnexpectedResponse(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::Request(msg) => write!(f, "couldn't reach Ollama: {msg}"),
            EmbedError::UnexpectedResponse(msg) => write!(f, "unexpected response from Ollama: {msg}"),
        }
    }
}

impl std::error::Error for EmbedError {}

// ── Wire format ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f64>>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Embeds a single piece of text via Ollama -- the direct equivalent of
/// the Python prototype's get_embedding(text): one string in, one vector
/// out, no batching, no caching. Whether a given raw_term actually needs
/// embedding at all (vs. reusing a cached one from a prior run) is a
/// decision for whatever calls this, not this function's job.
pub fn embed(text: &str, config: &EmbedConfig) -> Result<Vec<f64>, EmbedError> {
    let client = reqwest::blocking::Client::new();
    let url = format!("{}/api/embed", config.base_url);

    let response = client
        .post(&url)
        .json(&EmbedRequest {
            model: &config.model,
            input: text,
        })
        .send()
        .map_err(|e| EmbedError::Request(e.to_string()))?;

    let body: EmbedResponse = response
        .json()
        .map_err(|e| EmbedError::UnexpectedResponse(e.to_string()))?;

    parse_embedding(body)
}

/// Split out from embed() deliberately -- this is the part that's pure
/// and testable without a live Ollama instance. embed() itself is a thin
/// I/O wrapper around this.
fn parse_embedding(body: EmbedResponse) -> Result<Vec<f64>, EmbedError> {
    body.embeddings
        .into_iter()
        .next()
        .ok_or_else(|| EmbedError::UnexpectedResponse("response contained no embeddings".into()))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serializes_to_expected_shape() {
        let req = EmbedRequest {
            model: "nomic-embed-text",
            input: "hyperbolic geometry",
        };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"model":"nomic-embed-text","input":"hyperbolic geometry"}"#);
    }

    #[test]
    fn test_parse_embedding_extracts_first_vector() {
        let body = EmbedResponse {
            embeddings: vec![vec![0.1, 0.2, 0.3]],
        };
        let result = parse_embedding(body).unwrap();
        assert_eq!(result, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_parse_embedding_fails_on_empty_response() {
        let body = EmbedResponse { embeddings: vec![] };
        assert!(parse_embedding(body).is_err());
    }

    #[test]
    fn test_deserializes_real_ollama_response_shape() {
        // Matches the documented /api/embed response shape exactly --
        // guards against a mismatch between what this file assumes and
        // what Ollama actually sends.
        let raw = r#"{"embeddings":[[0.1,-0.2,0.3]]}"#;
        let body: EmbedResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(body.embeddings, vec![vec![0.1, -0.2, 0.3]]);
    }

    #[test]
    fn test_default_config_targets_nomic_embed_text() {
        let config = EmbedConfig::default();
        assert_eq!(config.model, "nomic-embed-text");
        assert_eq!(config.base_url, "http://localhost:11434");
    }
}