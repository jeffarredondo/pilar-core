// Drop this in examples/dump_terms.rs and run:
//   cargo run --release --example dump_terms -- docs/spacex_s1.txt
//
// The first pass (top-scored terms) doesn't tell you whether spaCy
// contributed anything -- min_occurrences and max_concepts_per_source
// can both throw away a real entity before it ever shows up in that
// list. This instead compares n-gram-only extraction against the
// hybrid extraction, chunk by chunk, over the WHOLE corpus, and prints
// every term spaCy found that the n-gram side didn't -- the actual
// question we care about, independent of TF-IDF filtering.

use std::collections::HashSet;
use std::path::PathBuf;

use pilar_core::extraction;
use pilar_core::ingest::{self, IngestConfig};
use pilar_core::ngram;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: dump_terms <source_file>...");
        std::process::exit(1);
    }

    let hybrid = extraction::build_hybrid_extractor();
    println!("spaCy available: {}\n", hybrid.spacy_available());

    let ngram_extractor = ngram::build_extractor();

    let mut ngram_only_terms: HashSet<String> = HashSet::new();
    let mut hybrid_terms: HashSet<String> = HashSet::new();
    let mut total_chunks = 0usize;

    for path in args.iter().map(PathBuf::from) {
        let chunks = match ingest::chunk_file(&path, &IngestConfig::default()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("failed to read {}: {e}", path.display());
                std::process::exit(1);
            }
        };
        total_chunks += chunks.len();

        for chunk in &chunks {
            ngram_only_terms.extend(ngram::extract_terms(&chunk.text, &ngram_extractor));
            hybrid_terms.extend(extraction::extract_terms(&chunk.text, &hybrid));
        }
    }

    let spacy_only: Vec<&String> = hybrid_terms.difference(&ngram_only_terms).collect();

    println!("{total_chunks} chunks");
    println!("n-gram-only unique terms:  {}", ngram_only_terms.len());
    println!("hybrid unique terms:       {}", hybrid_terms.len());
    println!("terms ONLY spaCy found:    {}\n", spacy_only.len());

    let mut sorted_spacy_only = spacy_only.clone();
    sorted_spacy_only.sort();

    println!("--- terms spaCy contributed that n-grams alone missed ---");
    for term in &sorted_spacy_only {
        println!("{term}");
    }
}