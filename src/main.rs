use std::path::PathBuf;
use std::process::ExitCode;

use pilar_core::pipeline::{self, PipelineConfig};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        eprintln!("usage: pilar-core <source_file>... [--output <dir>] [--dry-run]");
        eprintln!();
        eprintln!("  Ingests one or more text files, builds the knowledge manifold,");
        eprintln!("  and writes root.km / periphery-N.km / registry.km to the output dir.");
        eprintln!("  Output dir defaults to ./km_output. Requires Ollama running locally");
        eprintln!("  with nomic-embed-text, tinyllama, and mistral pulled.");
        eprintln!();
        eprintln!("  --dry-run: score the corpus and print how many terms survive at");
        eprintln!("  various strength_threshold values. No Ollama calls, no output files --");
        eprintln!("  just real numbers to pick a threshold from before a full run.");
        return ExitCode::FAILURE;
    }

    let mut source_paths = Vec::new();
    let mut output_dir = PathBuf::from("./km_output");
    let mut dry_run = false;

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--output" {
            i += 1;
            match args.get(i) {
                Some(dir) => output_dir = PathBuf::from(dir),
                None => {
                    eprintln!("error: --output requires a directory argument");
                    return ExitCode::FAILURE;
                }
            }
        } else if args[i] == "--dry-run" {
            dry_run = true;
        } else {
            source_paths.push(PathBuf::from(&args[i]));
        }
        i += 1;
    }

    if dry_run {
        return run_dry(&source_paths);
    }

    println!("Pilar — knowledge manifold pipeline");
    println!("  sources: {} file(s)", source_paths.len());
    for p in &source_paths {
        println!("    {}", p.display());
    }
    println!("  output: {}", output_dir.display());
    println!();

    // Load config from pilar.toml if present, fall back to default
    let config = if std::path::Path::new("pilar.toml").exists() {
        match pipeline::PipelineConfig::from_file(std::path::Path::new("pilar.toml")) {
            Ok(c) => {
                println!("  config: pilar.toml");
                // output_dir from CLI overrides config file
                pipeline::PipelineConfig { output_dir, ..c }
            }
            Err(e) => {
                eprintln!("warning: failed to load pilar.toml ({e}), using defaults");
                pipeline::PipelineConfig { output_dir, ..Default::default() }
            }
        }
    } else {
        pipeline::PipelineConfig { output_dir, ..Default::default() }
    };


    match pipeline::run(&source_paths, &config) {
        Ok(()) => {
            println!();
            println!("done.");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!();
            eprintln!("pipeline failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Scores the corpus and exits, no Ollama calls and no output files --
/// fast enough to run before every real attempt, so a bad
/// strength_threshold guess costs seconds, not another 40-minute
/// enrichment run.
fn run_dry(source_paths: &[PathBuf]) -> ExitCode {
    println!("dry run — scoring only, no Ollama calls, no files written");
    println!();

    match pipeline::ingest_and_score(source_paths, &PipelineConfig::default()) {
        Ok((chunks, scores)) => {
            println!("{} chunks, {} scored terms", chunks.len(), scores.len());
            println!();

            let thresholds = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9];
            let counts = pipeline::threshold_survival_counts(&scores, &thresholds);

            println!("strength_threshold -> terms surviving");
            for (t, count) in counts {
                println!("  {t:>6.3}  ->  {count}");
            }

            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("scoring failed: {e}");
            ExitCode::FAILURE
        }
    }
}