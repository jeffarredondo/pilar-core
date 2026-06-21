use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use crate::embed::{self, EmbedConfig, EmbedError};
use crate::enrich::{self, EnrichConfig, EnrichError};
use crate::ingest::{self, IngestConfig};
use crate::km::{self, KmError};
use crate::ner;
use crate::placement::{self, EmbeddedTerm, PlacementConfig, PlacementResult};
use crate::sharding::ShardRegistry;
use crate::tfidf::{self, ScoredTerm, TfidfConfig};
use crate::types::Chunk;

// ── Config ────────────────────────────────────────────────────────────────────

pub struct PipelineConfig {
    pub ingest: IngestConfig,
    pub tfidf: TfidfConfig,
    pub embed: EmbedConfig,
    pub placement: PlacementConfig,
    pub enrich: EnrichConfig,
    pub output_dir: PathBuf,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            ingest: IngestConfig::default(),
            tfidf: TfidfConfig::default(),
            embed: EmbedConfig::default(),
            placement: PlacementConfig::default(),
            enrich: EnrichConfig::default(),
            output_dir: PathBuf::from("."),
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum PipelineError {
    Io(std::io::Error),
    Embed(EmbedError),
    Enrich(EnrichError),
    Km(KmError),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Io(e) => write!(f, "IO error: {e}"),
            PipelineError::Embed(e) => write!(f, "embedding error: {e}"),
            PipelineError::Enrich(e) => write!(f, "enrichment error: {e}"),
            PipelineError::Km(e) => write!(f, "storage error: {e}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<std::io::Error> for PipelineError {
    fn from(e: std::io::Error) -> Self {
        PipelineError::Io(e)
    }
}
impl From<EmbedError> for PipelineError {
    fn from(e: EmbedError) -> Self {
        PipelineError::Embed(e)
    }
}
impl From<EnrichError> for PipelineError {
    fn from(e: EnrichError) -> Self {
        PipelineError::Enrich(e)
    }
}
impl From<KmError> for PipelineError {
    fn from(e: KmError) -> Self {
        PipelineError::Km(e)
    }
}

// ── Stage 1: ingest + extract + score ─────────────────────────────────────────

/// Chunks every source file, extracts candidate terms per chunk, and
/// scores them across the full corpus -- ingest.rs, ner.rs, and tfidf.rs
/// chained in the order each was built against. Only I/O here is reading
/// the source files themselves; everything past that is pure.
pub fn ingest_and_score(source_paths: &[PathBuf], config: &PipelineConfig) -> Result<(Vec<Chunk>, Vec<ScoredTerm>), PipelineError> {
    let extractor = ner::build_extractor();

    let mut all_chunks = Vec::new();
    for path in source_paths {
        let chunks = ingest::chunk_file(path, &config.ingest)?;
        all_chunks.extend(chunks);
    }

    let terms_per_chunk: Vec<HashSet<String>> = all_chunks.iter().map(|c| ner::extract_terms(&c.text, &extractor)).collect();

    let scores = tfidf::compute(&all_chunks, &terms_per_chunk, &config.tfidf);

    Ok((all_chunks, scores))
}

// ── Stage 2: embedding (real I/O -- not part of the testable core) ───────────

/// Embeds every scored term via Ollama. The one stage with no honest
/// local stand-in -- there's no way to fake what a real embedding model
/// produces, so this isn't covered by cargo test, the same boundary
/// embed.rs itself draws between embed() and parse_embedding().
///
/// Aborts on the first failure rather than skipping and continuing --
/// matches the Python prototype's behavior (get_embedding had no retry
/// or partial-failure handling either), not a robustness gap introduced
/// here. Worth revisiting if a real run hits it, not before.
pub fn embed_terms(scores: &[ScoredTerm], config: &EmbedConfig) -> Result<HashMap<String, Vec<f64>>, PipelineError> {
    let mut embeddings = HashMap::with_capacity(scores.len());
    for term in scores {
        let vec = embed::embed(&term.term, config)?;
        embeddings.insert(term.term.clone(), vec);
    }
    Ok(embeddings)
}

// ── Stage 3: placement (pure given embeddings) ────────────────────────────────

/// Places already-scored terms on the manifold, given their embeddings.
/// Takes embeddings as a plain lookup instead of computing them itself
/// -- that's what keeps this testable without a live Ollama: a test can
/// hand it any HashMap of the right shape, production code hands it
/// embed_terms' real output. A term with no matching embedding is
/// skipped rather than panicking -- defensive for this function used
/// standalone; unreachable in the real pipeline, since embed_terms
/// aborts entirely on any single failure rather than returning partial
/// results.
pub fn place_corpus(
    scores: &[ScoredTerm],
    embeddings: &HashMap<String, Vec<f64>>,
    config: &PlacementConfig,
    registry: &mut ShardRegistry,
) -> PlacementResult {
    let strengths = tfidf::normalize_to_strength(scores);

    let terms: Vec<EmbeddedTerm> = scores
        .iter()
        .filter_map(|s| {
            embeddings.get(&s.term).map(|emb| EmbeddedTerm {
                term: s.term.clone(),
                strength: *strengths.get(&s.term).unwrap_or(&0.0),
                embedding: emb.clone(),
                source_path: s.source_path.clone(),
                source_line: s.source_line,
            })
        })
        .collect();

    placement::place(terms, config, registry)
}

// ── Stage 4: enrichment (real I/O -- not part of the testable core) ──────────

/// Runs both enrichment steps over every placed concept, in place. Real
/// I/O (two Ollama chat calls per concept), so not unit-tested here for
/// the same reason embedding isn't -- this is the stage expected to
/// dominate wall-clock time, per the original concern that motivated
/// adding timing instrumentation to this file in the first place.
pub fn enrich_all(
    result: &mut PlacementResult,
    all_chunks: &[Chunk],
    all_terms: &[ScoredTerm],
    strengths: &HashMap<String, f64>,
    config: &EnrichConfig,
) -> Result<(), PipelineError> {
    let chunk_indices_by_term: HashMap<&str, &[usize]> =
        all_terms.iter().map(|t| (t.term.as_str(), t.chunk_indices.as_slice())).collect();

    let total = result.root.len() + result.periphery.values().map(|v| v.len()).sum::<usize>();
    let all_concepts = result.root.iter_mut().chain(result.periphery.values_mut().flatten());

    for (i, concept) in all_concepts.enumerate() {
        let indices = chunk_indices_by_term.get(concept.raw_term.as_str()).copied().unwrap_or(&[]);
        enrich::enrich_concept(concept, indices, all_chunks, all_terms, strengths, config)?;
        print_progress(i + 1, total);
    }
    if total > 0 {
        println!(); // move past the progress line once the batch finishes
    }

    Ok(())
}

/// Bare carriage-return progress bar -- no new dependency for something
/// this simple. Overwrites the same terminal line each call; negligible
/// cost next to the Ollama round trip each iteration is already waiting on.
fn print_progress(current: usize, total: usize) {
    let width = 30;
    let ratio = current as f64 / total.max(1) as f64;
    let filled = ((width as f64) * ratio).round() as usize;
    let filled = filled.min(width);
    let bar: String = "█".repeat(filled) + &"░".repeat(width - filled);
    print!("\r  enriching: [{bar}] {current}/{total}");
    let _ = std::io::stdout().flush();
}

// ── Stage 5: persistence ──────────────────────────────────────────────────────

/// Writes every shard to disk, plus the registry snapshot needed to
/// resume routing into the same periphery shards next run.
pub fn write_all(result: &PlacementResult, registry: &ShardRegistry, config: &PipelineConfig) -> Result<(), PipelineError> {
    std::fs::create_dir_all(&config.output_dir)?;

    km::write_shard(&result.root, "root", &config.output_dir.join("root.km"))?;

    for (shard_id, concepts) in &result.periphery {
        let path = config.output_dir.join(format!("{shard_id}.km"));
        km::write_shard(concepts, shard_id, &path)?;
    }

    km::write_registry(registry, &config.output_dir.join("registry.km"))?;

    Ok(())
}

// ── Full run ──────────────────────────────────────────────────────────────────

/// Runs every stage end to end: ingest -> extract -> score -> embed ->
/// place -> enrich -> write. The only function in this file that touches
/// a live Ollama for both embedding and enrichment, so it's genuinely
/// outside cargo test's reach -- validating it works means running it
/// against a real corpus, not a unit test. Every stage it calls into has
/// already been tested on its own, pure-logic terms; this is just the
/// wiring, timed per stage since enrichment (the ~120 Ollama calls) was
/// always expected to dominate wall-clock time, not NER or chunking.
pub fn run(source_paths: &[PathBuf], config: &PipelineConfig) -> Result<(), PipelineError> {
    let t0 = Instant::now();
    let (chunks, scores) = ingest_and_score(source_paths, config)?;
    println!("ingest+extract+score: {:?} ({} chunks, {} terms)", t0.elapsed(), chunks.len(), scores.len());

    let strengths = tfidf::normalize_to_strength(&scores);

    let t1 = Instant::now();
    let embeddings = embed_terms(&scores, &config.embed)?;
    println!("embed: {:?} ({} embeddings)", t1.elapsed(), embeddings.len());

    let t2 = Instant::now();
    let mut registry = ShardRegistry::new();
    let mut result = place_corpus(&scores, &embeddings, &config.placement, &mut registry);
    println!(
        "place: {:?} ({} root, {} periphery shards)",
        t2.elapsed(),
        result.root.len(),
        result.periphery.len()
    );

    let t3 = Instant::now();
    enrich_all(&mut result, &chunks, &scores, &strengths, &config.enrich)?;
    println!("enrich: {:?}", t3.elapsed());

    let t4 = Instant::now();
    write_all(&result, &registry, config)?;
    println!("write: {:?}", t4.elapsed());

    println!("total: {:?}", t0.elapsed());

    Ok(())
}

/// For each candidate threshold, how many terms would survive at that
/// strength_threshold. No placement, no Ollama -- just the scoring math,
/// so it's near-instant and safe to run before committing to a full
/// pipeline run. Built directly off the first real run putting 1747
/// terms through enrichment from a single 64KB file: strength_threshold
/// was always flagged as an unverified guess (0.001, far more permissive
/// than Python's min_tf=10 ever was), and this is what finally gives a
/// real, corpus-specific number to react to instead of guessing again.
pub fn threshold_survival_counts(scores: &[ScoredTerm], thresholds: &[f64]) -> Vec<(f64, usize)> {
    let strengths = tfidf::normalize_to_strength(scores);
    thresholds.iter().map(|&t| (t, strengths.values().filter(|&&s| s >= t).count())).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("pilar_test_pipeline_{label}_{}.txt", std::process::id()))
    }

    #[test]
    fn test_ingest_and_score_wires_chunking_through_to_scoring() {
        let path = temp_path("ingest");
        std::fs::write(
            &path,
            "Hyperbolic geometry and manifold structure repeat across this text. \
             Hyperbolic geometry is genuinely everywhere in this corpus.",
        )
        .unwrap();

        let config = PipelineConfig {
            ingest: IngestConfig {
                chunk_size: 50,
                overlap: 5,
            },
            // Permissive on purpose -- this test checks that chunking
            // wires through to scoring correctly, not whether a small
            // hand-written fixture happens to hit a realistic
            // min_occurrences. The default (10) is tuned for real
            // corpora, not test prose.
            tfidf: TfidfConfig { min_occurrences: 1 },
            ..Default::default()
        };

        let (chunks, scores) = ingest_and_score(&[path.clone()], &config).unwrap();

        assert!(!chunks.is_empty());
        assert!(!scores.is_empty());
        assert!(chunks.iter().all(|c| c.source_path == path));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_ingest_and_score_reports_io_error_for_missing_file() {
        let result = ingest_and_score(&[PathBuf::from("/nonexistent/source.txt")], &PipelineConfig::default());
        assert!(matches!(result, Err(PipelineError::Io(_))));
    }

    #[test]
    fn test_place_corpus_wires_scores_through_to_a_nonempty_result() {
        let path = temp_path("place");
        std::fs::write(
            &path,
            "Knowledge graphs and retrieval systems are discussed at length here. \
             Graph navigation differs meaningfully from plain retrieval.",
        )
        .unwrap();

        let config = PipelineConfig {
            ingest: IngestConfig {
                chunk_size: 50,
                overlap: 5,
            },
            // Same reasoning as the ingest test above -- this checks
            // wiring, not whether the production min_occurrences default
            // happens to suit a two-sentence fixture.
            tfidf: TfidfConfig { min_occurrences: 1 },
            ..Default::default()
        };

        let (_chunks, scores) = ingest_and_score(&[path.clone()], &config).unwrap();

        // Stand-in embeddings -- no live Ollama involved, just confirming
        // the wiring from scores through placement actually works.
        let embeddings: HashMap<String, Vec<f64>> = scores
            .iter()
            .enumerate()
            .map(|(i, s)| (s.term.clone(), vec![i as f64 * 0.1, 0.2, 0.3, 0.4]))
            .collect();

        let mut registry = ShardRegistry::new();
        let result = place_corpus(&scores, &embeddings, &config.placement, &mut registry);

        let total_placed = result.root.len() + result.periphery.values().map(|v| v.len()).sum::<usize>();
        assert!(total_placed > 0);
        assert_eq!(total_placed, scores.len(), "every scored term had an embedding, so none should be skipped");

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_place_corpus_skips_terms_with_no_embedding_instead_of_panicking() {
        let scores = vec![ScoredTerm {
            term: "orphaned".to_string(),
            tfidf: 1.0,
            chunk_indices: vec![0],
            source_path: PathBuf::from("t.txt"),
            source_line: None,
        }];
        let embeddings = HashMap::new(); // deliberately empty

        let mut registry = ShardRegistry::new();
        let result = place_corpus(&scores, &embeddings, &PlacementConfig::default(), &mut registry);

        let total_placed = result.root.len() + result.periphery.values().map(|v| v.len()).sum::<usize>();
        assert_eq!(total_placed, 0);
    }

    #[test]
    fn test_threshold_survival_counts_decreases_as_threshold_rises() {
        let scores = vec![
            ScoredTerm {
                term: "weak".to_string(),
                tfidf: 0.1,
                chunk_indices: vec![0],
                source_path: PathBuf::from("t.txt"),
                source_line: None,
            },
            ScoredTerm {
                term: "strong".to_string(),
                tfidf: 10.0,
                chunk_indices: vec![0],
                source_path: PathBuf::from("t.txt"),
                source_line: None,
            },
        ];

        let counts = threshold_survival_counts(&scores, &[0.001, 0.5, 0.99]);

        assert_eq!(counts[0].1, 2, "near-zero threshold should keep both terms");
        assert!(counts[2].1 <= counts[0].1, "higher threshold should never keep more terms");
        assert!(counts[2].1 >= 1, "the single strongest term should survive even a near-max threshold");
    }

    #[test]
    fn test_write_all_produces_root_and_registry_files() {
        let dir = std::env::temp_dir().join(format!("pilar_test_write_all_{}", std::process::id()));

        let scores = vec![ScoredTerm {
            term: "manifold".to_string(),
            tfidf: 1.0,
            chunk_indices: vec![0],
            source_path: PathBuf::from("t.txt"),
            source_line: None,
        }];
        let mut embeddings = HashMap::new();
        embeddings.insert("manifold".to_string(), vec![0.1, 0.2, 0.3, 0.4]);

        let config = PipelineConfig {
            output_dir: dir.clone(),
            ..Default::default()
        };

        let mut registry = ShardRegistry::new();
        let result = place_corpus(&scores, &embeddings, &config.placement, &mut registry);

        write_all(&result, &registry, &config).unwrap();

        assert!(dir.join("root.km").exists());
        assert!(dir.join("registry.km").exists());

        std::fs::remove_dir_all(dir).ok();
    }
}