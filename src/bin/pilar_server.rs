//! pilar-server -- MCP/HTTP wrapper around the pipeline in this crate.
//!
//! Exposes three tools over Streamable HTTP (per the Sep 27 2026 handoff
//! decisions):
//!   - `query`        read path: RAG answer + ranked concepts, same logic
//!                     as infer.rs's query loop.
//!   - `list_shards`  registry overview: shard ids, anchor positions.
//!   - `save_concept` write path: appends to a scratch corpus (JSONL) on
//!                     disk. Deliberately does NOT touch the pipeline --
//!                     no extraction, no TF-IDF, no embedding, no shard
//!                     write. The scratch corpus is promoted into the live
//!                     manifold later by a separate batch-ingest step
//!                     (not built yet -- see the handoff doc), which is
//!                     expected to skip the TF-IDF occurrence floor for
//!                     scratch-origin batches specifically.
//!
//! These tools are meant to sit behind an OpenClaw Skill, not be handed to
//! agents directly -- the Skill is where "should I even query" and "is
//! this worth saving" judgment calls live. See the handoff doc for why.
//!
//! Run with: cargo run --bin pilar-server
//! Config:   PILAR_KM_DIR (default ./km_output), PILAR_BIND (default
//!           127.0.0.1:8090)

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars,
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde::Deserialize;
use serde_json::json;

use pilar_core::embed::{self, EmbedConfig};
use pilar_core::enrich::{self, EnrichConfig};
use pilar_core::geometry::poincare_distance;
use pilar_core::km::{read_registry, read_shard};
use pilar_core::placement::{PlacementConfig, Projections};
use pilar_core::types::ManifoldCoord;

// ── Request shapes ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct QueryRequest {
    /// Natural-language question to answer using only facts already in the
    /// manifold. Only call this for questions that plausibly need
    /// corpus-grounded facts (named entities, dates, dollar figures) --
    /// not for general conversation, opinions, or things answerable
    /// without lookup.
    question: String,
    /// How many nearest shards to load before ranking concepts within
    /// them. Defaults to 6.
    #[serde(default = "default_shard_count")]
    shard_count: usize,
    /// How many top-ranked concepts (by Poincare distance to the query)
    /// to feed into the answer prompt. Defaults to 5.
    #[serde(default = "default_top_k")]
    top_k: usize,
}
fn default_shard_count() -> usize {
    6
}
fn default_top_k() -> usize {
    5
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SaveConceptRequest {
    /// The fact or note to remember. Appended to a scratch corpus, not
    /// the live manifold -- it only becomes queryable after a later batch
    /// ingest promotes it. Save one clear, self-contained fact per call
    /// rather than a running log entry.
    text: String,
    /// Optional free-form tag for grouping or filtering scratch notes
    /// later (e.g. a topic or source conversation id).
    #[serde(default)]
    tag: Option<String>,
}

// ── Server ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct PilarServer {
    km_dir: PathBuf,
    scratch_path: PathBuf,
    embed_config: Arc<EmbedConfig>,
    enrich_config: Arc<EnrichConfig>,
    projection_seed: u64,
    tool_router: ToolRouter<PilarServer>,
}

impl PilarServer {
    fn new(km_dir: PathBuf, scratch_path: PathBuf) -> Self {
        // Matches infer.rs: EmbedConfig/EnrichConfig::default(), not
        // pilar.toml -- infer.rs doesn't read pilar.toml either, so this
        // preserves the existing (if a little surprising) pattern rather
        // than silently diverging from the reference query path. Worth
        // revisiting together if that's unwanted.
        Self::with_config(km_dir, scratch_path, EmbedConfig::default(), EnrichConfig::default())
    }

    /// Split out from new() so tests can point embed_config at a
    /// deliberately-dead address instead of relying on Ollama actually
    /// being absent from the test machine -- the first version of the
    /// Ollama-unreachable test assumed that and failed on a dev machine
    /// that had Ollama running locally, which is exactly the flakiness
    /// this constructor exists to avoid.
    fn with_config(km_dir: PathBuf, scratch_path: PathBuf, embed_config: EmbedConfig, enrich_config: EnrichConfig) -> Self {
        Self {
            km_dir,
            scratch_path,
            embed_config: Arc::new(embed_config),
            enrich_config: Arc::new(enrich_config),
            projection_seed: PlacementConfig::default().projection_seed,
            tool_router: Self::tool_router(),
        }
    }
}

// ── query: pure logic, ported from infer.rs ──────────────────────────────────

fn build_rag_prompt(query: &str, concepts: &[(f64, String, String, String)]) -> String {
    let context = concepts
        .iter()
        .map(|(_, raw_term, _, desc)| format!("- {raw_term}: {desc}"))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "Answer the following question using only the facts provided below.\n\
Do not use outside knowledge. If the facts do not contain enough information, say so.\n\
\n\
FACTS:\n\
{context}\n\
\n\
QUESTION: {query}\n\
\n\
ANSWER:"
    )
}

/// Runs the same embed -> project -> nearest-shards -> rank -> RAG-answer
/// flow as infer.rs's query loop, parameterized instead of hardcoded, and
/// returning structured JSON instead of printing. Blocking (reqwest
/// blocking client underneath embed::embed/enrich::chat) -- callers must
/// run this via spawn_blocking, not directly on the async runtime.
fn run_query(
    km_dir: &Path,
    embed_config: &EmbedConfig,
    enrich_config: &EnrichConfig,
    projection_seed: u64,
    req: &QueryRequest,
) -> Result<serde_json::Value, String> {
    let registry_path = km_dir.join("registry.km");
    let registry = read_registry(&registry_path).map_err(|e| format!("failed to read registry: {e}"))?;

    let query_embedding = embed::embed(&req.question, embed_config).map_err(|e| format!("embed failed: {e}"))?;

    let projections = Projections::new(query_embedding.len(), projection_seed);
    let query_dir = projections.hyperbolic_direction(&query_embedding);
    let query_pos = [query_dir[0] * 0.5, query_dir[1] * 0.5, query_dir[2] * 0.5];

    let nearest_shards = registry.nearest_shards(&query_pos, req.shard_count);

    let mut loaded_shards = Vec::new();
    for (anchor, _) in &nearest_shards {
        let path = km_dir.join(format!("{}.km", anchor.shard_id));
        if let Ok(shard) = read_shard(&path) {
            loaded_shards.push(shard);
        }
    }

    let mut ranked: Vec<(f64, String, String, String)> = Vec::new();
    for shard in &loaded_shards {
        for (_, concept) in &shard.concepts {
            if let ManifoldCoord::Hyperbolic { position } = &concept.coordinate {
                let dist = poincare_distance(&query_pos, position);
                ranked.push((dist, concept.raw_term.clone(), concept.label.clone(), concept.description.clone()));
            }
        }
    }
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    ranked.truncate(req.top_k);

    // Full, untruncated descriptions go into the prompt -- see pipeline.rs
    // history on why truncating here (rather than only at print time) was
    // the bug that silently starved the RAG prompt of facts.
    let prompt = build_rag_prompt(&req.question, &ranked);
    let answer = enrich::chat(&prompt, &enrich_config.summarize_model, enrich_config)
        .map_err(|e| format!("answer generation failed: {e}"))?;

    let concepts_json: Vec<_> = ranked
        .iter()
        .map(|(dist, raw_term, label, desc)| {
            json!({
                "distance": dist,
                "raw_term": raw_term,
                "label": label,
                "description": desc,
            })
        })
        .collect();

    let shards_searched: Vec<_> = nearest_shards
        .iter()
        .map(|(a, d)| json!({ "shard_id": a.shard_id, "distance": d }))
        .collect();

    Ok(json!({
        "question": req.question,
        "answer": answer,
        "concepts": concepts_json,
        "shards_searched": shards_searched,
    }))
}

// ── save_concept: pure I/O, no pipeline involvement ──────────────────────────

/// Appends one JSONL record to the scratch corpus. Deliberately does not
/// touch extraction, TF-IDF, embedding, or shard writing -- see the
/// module doc comment for why. `source: "scratch"` is the provenance tag
/// the eventual batch-ingest step relies on to skip the min_occurrences
/// floor for this batch only.
fn append_scratch(path: &Path, req: &SaveConceptRequest) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let saved_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let record = json!({
        "text": req.text,
        "tag": req.tag,
        "source": "scratch",
        "saved_at": saved_at,
    });

    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    writeln!(file, "{record}").map_err(|e| e.to_string())?;
    Ok(())
}

// ── Tools ─────────────────────────────────────────────────────────────────────

#[tool_router]
impl PilarServer {
    #[tool(
        description = "Answer a natural-language question using facts already placed in the knowledge manifold (RAG over ingested corpora). Only call this when the question plausibly needs corpus-grounded facts -- not for general conversation."
    )]
    async fn query(&self, Parameters(req): Parameters<QueryRequest>) -> Result<CallToolResult, McpError> {
        let km_dir = self.km_dir.clone();
        let embed_config = self.embed_config.clone();
        let enrich_config = self.enrich_config.clone();
        let projection_seed = self.projection_seed;

        // Only a genuine task panic (server-side bug) becomes a hard
        // protocol-level McpError here. An ordinary failure in the query
        // itself -- registry missing, Ollama unreachable, embed/chat
        // erroring -- is expected/recoverable and reported as a soft
        // CallToolResult error instead, so the calling model actually
        // sees the message and can reason about or retry it, rather than
        // the call just failing outright with no readable content.
        let outcome = tokio::task::spawn_blocking(move || {
            run_query(&km_dir, &embed_config, &enrich_config, projection_seed, &req)
        })
        .await
        .map_err(|e| McpError::internal_error(format!("query task panicked: {e}"), None))?;

        match outcome {
            Ok(result) => Ok(CallToolResult::success(vec![ContentBlock::text(result.to_string())])),
            Err(msg) => Ok(CallToolResult::error(vec![ContentBlock::text(msg)])),
        }
    }

    #[tool(description = "List the manifold's shard registry: shard ids, anchor positions, and total shard count. Cheap, no Ollama calls.")]
    async fn list_shards(&self) -> Result<CallToolResult, McpError> {
        let km_dir = self.km_dir.clone();

        let outcome = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, String> {
            let registry_path = km_dir.join("registry.km");
            let registry = read_registry(&registry_path).map_err(|e| format!("failed to read registry: {e}"))?;
            let shards: Vec<_> = registry
                .anchors()
                .iter()
                .map(|a| json!({ "shard_id": a.shard_id, "position": a.position }))
                .collect();
            Ok(json!({ "shard_count": registry.anchor_count(), "shards": shards }))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("list_shards task panicked: {e}"), None))?;

        match outcome {
            Ok(result) => Ok(CallToolResult::success(vec![ContentBlock::text(result.to_string())])),
            Err(msg) => Ok(CallToolResult::error(vec![ContentBlock::text(msg)])),
        }
    }

    #[tool(
        description = "Save a fact or note to the scratch corpus for later promotion into the manifold. Does NOT make it queryable immediately -- scratch is only ingested once it crosses a token-count threshold. Use for durable, self-contained facts worth remembering across sessions, not routine chatter."
    )]
    async fn save_concept(&self, Parameters(req): Parameters<SaveConceptRequest>) -> Result<CallToolResult, McpError> {
        let scratch_path = self.scratch_path.clone();

        let outcome = tokio::task::spawn_blocking(move || append_scratch(&scratch_path, &req))
            .await
            .map_err(|e| McpError::internal_error(format!("save_concept task panicked: {e}"), None))?;

        match outcome {
            Ok(()) => Ok(CallToolResult::success(vec![ContentBlock::text("saved to scratch")])),
            Err(msg) => Ok(CallToolResult::error(vec![ContentBlock::text(msg)])),
        }
    }
}

#[tool_handler]
impl ServerHandler for PilarServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
            .with_protocol_version(ProtocolVersion::V_2024_11_05)
            .with_instructions(
                "Pilar knowledge-manifold server. Tools: query (RAG answer + ranked concepts), \
                 list_shards (registry overview), save_concept (write to scratch corpus, promoted later). \
                 Intended to be called from behind a gating Skill, not directly by every agent."
                    .to_string(),
            )
    }
}

// ── router construction (shared by main() and the integration tests) ────────

/// Wires a PilarServer factory into the actual axum router serving
/// `/mcp`. The StreamableHttpServerConfig here is the one thing that must
/// never drift between production and tests: the bug that caused
/// OpenClaw's client to hang for 30s on every call (rmcp defaulting to
/// SSE-framed responses its bundled client apparently couldn't parse)
/// lived entirely in this config, not in any tool's logic. Separated from
/// build_router() below so tests can vary *how a PilarServer gets built*
/// (e.g. pointing embed_config at a deliberately-dead address) while
/// still going through this exact same transport wiring.
fn build_router_with_factory(
    service_factory: impl Fn() -> Result<PilarServer, std::io::Error> + Send + Sync + 'static,
    ct: tokio_util::sync::CancellationToken,
) -> axum::Router {
    // LocalSessionManager stays -- `initialize` needs a real session
    // manager or it fails outright (confirmed: NeverSessionManager errors
    // "Session management is not supported" on the very first request).
    // The actual fix is with_legacy_session_mode(false): per rmcp's own
    // doc comment on StreamableHttpServerConfig::json_response, that flag
    // only takes effect when legacy_session_mode is false -- with the
    // default (true), every response is SSE-framed regardless of
    // json_response, which is what caused OpenClaw's bundled MCP client
    // to hang: server-side logs proved it answered ListToolsRequest in
    // <1ms, but the client timed out 30s later anyway, seemingly unable
    // to parse the SSE stream. legacy_session_mode(false) routes response
    // *serving* through the stateless/json_response-respecting path even
    // though LocalSessionManager still handles the initialize handshake.
    let service = StreamableHttpService::new(
        service_factory,
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default()
            .with_cancellation_token(ct.child_token())
            .with_legacy_session_mode(false)
            .with_json_response(true),
    );

    axum::Router::new().nest_service("/mcp", service)
}

/// Production entry point: builds a router whose PilarServer always uses
/// real, default Ollama config (EmbedConfig::default() etc, via
/// PilarServer::new()). main() and the ordinary integration tests use
/// this; only the one test that specifically exercises the
/// Ollama-unreachable path calls build_router_with_factory() directly
/// with a custom factory instead.
fn build_router(km_dir: PathBuf, scratch_path: PathBuf, ct: tokio_util::sync::CancellationToken) -> axum::Router {
    build_router_with_factory(move || Ok(PilarServer::new(km_dir.clone(), scratch_path.clone())), ct)
}

// ── main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber_init();

    let km_dir = PathBuf::from(std::env::var("PILAR_KM_DIR").unwrap_or_else(|_| "./km_output".to_string()));
    let bind_address = std::env::var("PILAR_BIND").unwrap_or_else(|_| "127.0.0.1:8090".to_string());
    let scratch_path = km_dir.join("scratch.jsonl");

    if !km_dir.join("registry.km").exists() {
        eprintln!(
            "warning: {} has no registry.km yet -- query/list_shards will fail until a corpus is ingested",
            km_dir.display()
        );
    }

    println!("pilar-server");
    println!("  km_dir: {}", km_dir.display());
    println!("  scratch: {}", scratch_path.display());
    println!("  bind: {bind_address}");
    println!("  mcp endpoint: http://{bind_address}/mcp");

    let ct = tokio_util::sync::CancellationToken::new();
    let router = build_router(km_dir, scratch_path, ct.clone());
    let listener = tokio::net::TcpListener::bind(&bind_address).await?;

    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            tokio::signal::ctrl_c().await.ok();
            ct.cancel();
        })
        .await?;

    Ok(())
}

/// Minimal tracing init -- kept out of main() just so main() reads as
/// "load config, print it, serve" without a subscriber-builder block in
/// the middle. Falls back to "info" if RUST_LOG isn't set.
fn tracing_subscriber_init() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    let _ = tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .try_init();
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// Two kinds, matching the rest of this crate's convention (pure logic
// unit-tested, I/O tested against a real temp path) plus one kind this
// crate didn't have anywhere yet: real HTTP integration tests that spin
// up the actual server via build_router() and hit it with a real client.
// That last kind exists specifically because the OpenClaw hang was a
// transport-config bug (SSE vs JSON responses) that no amount of testing
// build_rag_prompt or append_scratch in isolation would ever have caught
// -- only a test that goes over the wire and checks Content-Type proves
// this stays fixed.
#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pilar_test_server_{label}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // ── build_rag_prompt: pure ───────────────────────────────────────────────

    #[test]
    fn test_build_rag_prompt_includes_question_and_facts() {
        let concepts = vec![
            (0.1, "spacex".to_string(), "SpaceX".to_string(), "a rocket company".to_string()),
            (0.2, "starlink".to_string(), "Starlink".to_string(), "a satellite network".to_string()),
        ];
        let prompt = build_rag_prompt("What is SpaceX?", &concepts);

        assert!(prompt.contains("QUESTION: What is SpaceX?"));
        assert!(prompt.contains("- spacex: a rocket company"));
        assert!(prompt.contains("- starlink: a satellite network"));
        assert!(prompt.contains("Do not use outside knowledge"));
    }

    #[test]
    fn test_build_rag_prompt_handles_no_concepts() {
        let prompt = build_rag_prompt("anything?", &[]);
        assert!(prompt.contains("QUESTION: anything?"));
        assert!(prompt.contains("FACTS:\n\n"), "empty concept list should still produce a valid (empty) facts section, not panic");
    }

    // ── append_scratch: real filesystem I/O ──────────────────────────────────

    #[test]
    fn test_append_scratch_writes_source_scratch_tag() {
        let dir = temp_dir("append_basic");
        let path = dir.join("scratch.jsonl");
        let req = SaveConceptRequest {
            text: "Q1 2026 revenue was $4,694 million, not a loss.".to_string(),
            tag: Some("spacex-financials".to_string()),
        };

        append_scratch(&path, &req).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let record: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(record["source"], "scratch");
        assert_eq!(record["tag"], "spacex-financials");
        assert_eq!(record["text"], "Q1 2026 revenue was $4,694 million, not a loss.");
        assert!(record["saved_at"].as_u64().unwrap() > 0);

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_append_scratch_appends_rather_than_overwrites() {
        let dir = temp_dir("append_multi");
        let path = dir.join("scratch.jsonl");

        append_scratch(&path, &SaveConceptRequest { text: "first note".to_string(), tag: None }).unwrap();
        append_scratch(&path, &SaveConceptRequest { text: "second note".to_string(), tag: None }).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2, "second save_concept call should append a new line, not replace the file");

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(first["text"], "first note");
        assert_eq!(second["text"], "second note");

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_append_scratch_creates_parent_directory() {
        let dir = temp_dir("append_mkdir");
        let path = dir.join("nested").join("scratch.jsonl");
        assert!(!path.parent().unwrap().exists());

        append_scratch(&path, &SaveConceptRequest { text: "note".to_string(), tag: None }).unwrap();

        assert!(path.exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_append_scratch_omits_tag_as_null_when_none() {
        let dir = temp_dir("append_no_tag");
        let path = dir.join("scratch.jsonl");

        append_scratch(&path, &SaveConceptRequest { text: "untagged note".to_string(), tag: None }).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let record: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert!(record["tag"].is_null());

        std::fs::remove_dir_all(dir).ok();
    }

    // ── Integration: real HTTP over the real router ──────────────────────────

    /// Writes a minimal but real registry.km -- same TOML shape km.rs
    /// actually reads -- so integration tests exercise real
    /// read_registry() rather than a mock.
    fn write_fixture_registry(km_dir: &Path, anchors: &[(&str, [f64; 3])]) {
        let mut toml = String::from("next_id = 99\nprojection_seed = 42\nembedding_dim = 4\n\n");
        for (shard_id, pos) in anchors {
            toml.push_str(&format!(
                "[[anchors]]\nshard_id = \"{shard_id}\"\nposition = [{}, {}, {}]\n\n",
                pos[0], pos[1], pos[2]
            ));
        }
        std::fs::write(km_dir.join("registry.km"), toml).unwrap();
    }

    /// Spins up build_router() on an OS-assigned free port, same
    /// construction main() uses. Returns the /mcp base URL and a
    /// cancellation token the caller uses to shut the server down at the
    /// end of the test, so tests don't leak background tasks.
    async fn spawn_test_server(km_dir: PathBuf) -> (String, tokio_util::sync::CancellationToken) {
        let scratch_path = km_dir.join("scratch.jsonl");
        let ct = tokio_util::sync::CancellationToken::new();
        let router = build_router(km_dir, scratch_path, ct.clone());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let serve_ct = ct.clone();

        tokio::spawn(async move {
            let _ = axum::serve(listener, router)
                .with_graceful_shutdown(async move { serve_ct.cancelled().await })
                .await;
        });

        (format!("http://{addr}/mcp"), ct)
    }

    async fn post_mcp(base_url: &str, body: serde_json::Value) -> reqwest::Response {
        reqwest::Client::new()
            .post(base_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(&body)
            .send()
            .await
            .unwrap()
    }

    fn initialize_body() -> serde_json::Value {
        json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "pilar-server-test", "version": "0"}
            }
        })
    }

    /// The actual regression test for the OpenClaw hang: asserts the
    /// response is plain JSON, not SSE. Before the legacy_session_mode(false)
    /// + json_response(true) fix, this would have failed with
    /// content-type: text/event-stream -- exactly the mismatch that (most
    /// likely) made OpenClaw's client hang for 30 seconds on every call
    /// despite the server answering instantly.
    #[tokio::test]
    async fn test_initialize_responds_as_plain_json_not_sse() {
        let km_dir = temp_dir("http_init");
        write_fixture_registry(&km_dir, &[("shard-0", [0.0, 0.0, 0.0])]);
        let (base_url, ct) = spawn_test_server(km_dir.clone()).await;

        let response = post_mcp(&base_url, initialize_body()).await;

        assert_eq!(response.status(), 200);
        let content_type = response.headers().get("content-type").unwrap().to_str().unwrap().to_string();
        assert!(
            content_type.starts_with("application/json"),
            "expected application/json, got '{content_type}' -- the SSE-hang regression is back if this ever fails"
        );

        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["result"]["serverInfo"]["name"], "rmcp");

        ct.cancel();
        std::fs::remove_dir_all(km_dir).ok();
    }

    #[tokio::test]
    async fn test_tools_list_exposes_all_three_tools() {
        let km_dir = temp_dir("http_tools_list");
        write_fixture_registry(&km_dir, &[("shard-0", [0.0, 0.0, 0.0])]);
        let (base_url, ct) = spawn_test_server(km_dir.clone()).await;

        post_mcp(&base_url, initialize_body()).await;
        let response = post_mcp(&base_url, json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})).await;

        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        let names: Vec<&str> = body["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();

        assert!(names.contains(&"query"));
        assert!(names.contains(&"list_shards"));
        assert!(names.contains(&"save_concept"));
        assert_eq!(names.len(), 3, "expected exactly these three tools, no more no less");

        ct.cancel();
        std::fs::remove_dir_all(km_dir).ok();
    }

    #[tokio::test]
    async fn test_list_shards_returns_real_registry_contents_over_mcp() {
        let km_dir = temp_dir("http_list_shards");
        write_fixture_registry(
            &km_dir,
            &[("shard-0", [0.0, 0.0, 0.0]), ("shard-1", [0.5, 0.1, -0.2]), ("shard-2", [-0.3, 0.4, 0.1])],
        );
        let (base_url, ct) = spawn_test_server(km_dir.clone()).await;

        post_mcp(&base_url, initialize_body()).await;
        let response = post_mcp(
            &base_url,
            json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_shards", "arguments": {}}}),
        )
        .await;

        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["result"]["isError"], false);

        // The tool's own result comes back as a JSON string inside a text
        // content block (that's the MCP content-block convention this
        // server uses) -- parse that inner string to check the real data.
        let inner_text = body["result"]["content"][0]["text"].as_str().unwrap();
        let inner: serde_json::Value = serde_json::from_str(inner_text).unwrap();

        assert_eq!(inner["shard_count"], 3);
        let shard_ids: Vec<&str> = inner["shards"].as_array().unwrap().iter().map(|s| s["shard_id"].as_str().unwrap()).collect();
        assert!(shard_ids.contains(&"shard-0"));
        assert!(shard_ids.contains(&"shard-1"));
        assert!(shard_ids.contains(&"shard-2"));

        ct.cancel();
        std::fs::remove_dir_all(km_dir).ok();
    }

    #[tokio::test]
    async fn test_list_shards_reports_error_when_registry_missing() {
        let km_dir = temp_dir("http_list_shards_missing");
        // Deliberately no write_fixture_registry() call -- no registry.km exists.
        let (base_url, ct) = spawn_test_server(km_dir.clone()).await;

        post_mcp(&base_url, initialize_body()).await;
        let response = post_mcp(
            &base_url,
            json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_shards", "arguments": {}}}),
        )
        .await;

        assert_eq!(response.status(), 200, "MCP reports tool errors in-band (isError:true), not as an HTTP error status");
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["result"]["isError"], true, "missing registry.km should surface as a tool-level error, not silently succeed with no data");

        ct.cancel();
        std::fs::remove_dir_all(km_dir).ok();
    }

    #[tokio::test]
    async fn test_save_concept_over_mcp_writes_real_scratch_file() {
        let km_dir = temp_dir("http_save_concept");
        write_fixture_registry(&km_dir, &[("shard-0", [0.0, 0.0, 0.0])]);
        let (base_url, ct) = spawn_test_server(km_dir.clone()).await;

        post_mcp(&base_url, initialize_body()).await;
        let response = post_mcp(
            &base_url,
            json!({
                "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": {
                    "name": "save_concept",
                    "arguments": {"text": "Q1 2026 revenue correction: $4,694 million.", "tag": "spacex-financials"}
                }
            }),
        )
        .await;

        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["result"]["isError"], false);

        // Prove the write actually landed on disk via the real MCP round
        // trip, not just that the HTTP call returned 200 -- this is the
        // same file save_concept() promised in its tool description.
        let scratch_content = std::fs::read_to_string(km_dir.join("scratch.jsonl")).unwrap();
        let record: serde_json::Value = serde_json::from_str(scratch_content.trim()).unwrap();
        assert_eq!(record["source"], "scratch");
        assert_eq!(record["tag"], "spacex-financials");
        assert_eq!(record["text"], "Q1 2026 revenue correction: $4,694 million.");

        ct.cancel();
        std::fs::remove_dir_all(km_dir).ok();
    }

    /// Binds an OS-assigned free port and immediately drops the listener,
    /// so the returned address is guaranteed to have nothing listening on
    /// it -- unlike assuming "the test machine has no Ollama running",
    /// which is false on any dev machine actually using this pipeline
    /// (caught exactly this way: this test originally pointed at
    /// EmbedConfig::default()'s localhost:11434 and passed in CI/sandbox
    /// environments with no Ollama, then failed on a real dev machine
    /// where Ollama was reachable and the call plausibly succeeded).
    async fn unreachable_local_url() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn test_query_reports_error_when_ollama_unreachable() {
        // Deliberately-dead embed endpoint (see unreachable_local_url) --
        // this asserts the tool fails cleanly and in-band rather than
        // hanging or panicking, which is the guarantee this server
        // actually needs to uphold when Ollama is down, not a test of RAG
        // answer quality itself, and not dependent on whether the machine
        // running this test happens to have Ollama up or not.
        let km_dir = temp_dir("http_query_no_ollama");
        write_fixture_registry(&km_dir, &[("shard-0", [0.0, 0.0, 0.0])]);
        let scratch_path = km_dir.join("scratch.jsonl");
        let dead_embed = EmbedConfig { base_url: unreachable_local_url().await, model: "nomic-embed-text".to_string() };

        let ct = tokio_util::sync::CancellationToken::new();
        let km_dir_for_factory = km_dir.clone();
        let router = build_router_with_factory(
            move || {
                Ok(PilarServer::with_config(
                    km_dir_for_factory.clone(),
                    scratch_path.clone(),
                    EmbedConfig { base_url: dead_embed.base_url.clone(), model: dead_embed.model.clone() },
                    EnrichConfig::default(),
                ))
            },
            ct.clone(),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let serve_ct = ct.clone();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).with_graceful_shutdown(async move { serve_ct.cancelled().await }).await;
        });
        let base_url = format!("http://{addr}/mcp");

        post_mcp(&base_url, initialize_body()).await;
        let response = post_mcp(
            &base_url,
            json!({
                "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                "params": {"name": "query", "arguments": {"question": "does this fail cleanly?"}}
            }),
        )
        .await;

        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["result"]["isError"], true, "query should report a clean tool-level error when Ollama is unreachable, not hang or panic");

        ct.cancel();
        std::fs::remove_dir_all(km_dir).ok();
    }
}