use std::path::PathBuf;

use pilar_core::km::{read_registry, read_shard};
use pilar_core::placement::{Projections, PlacementConfig};
use pilar_core::embed::{embed, EmbedConfig};
use pilar_core::geometry::poincare_distance;
use pilar_core::types::ManifoldCoord;

use pilar_core::enrich::chat;
use pilar_core::enrich::EnrichConfig;

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

fn main() {
    let km_dir = PathBuf::from("./km_output");
    let queries = vec![
        "What was SpaceX's revenue and loss in Q1 2026?",
        "What is SpaceX's valuation for the IPO?",
        "What happened with xAI and SpaceX?",
        "Is SpaceX a good investment given its losses?",
        "Has market mania affected SpaceX's valuation?",
    ];

    let registry_path = km_dir.join("registry.km");
    let registry = read_registry(&registry_path).expect("failed to read registry");
    println!("Registry loaded: {} shards", registry.anchor_count());
    println!();

    let embed_config = EmbedConfig::default();
    let projection_seed = PlacementConfig::default().projection_seed;

    for query in &queries {
        println!("Query: {query}");
        println!("{}", "─".repeat(60));

        let query_embedding = embed(query, &embed_config).expect("embed failed");

        // Option B: use projected direction only, no fixed radius
        // The query doesn't have a strength — don't assign one
        let projections = Projections::new(query_embedding.len(), projection_seed);
        let query_dir = projections.hyperbolic_direction(&query_embedding);
        let query_pos = [query_dir[0] * 0.5, query_dir[1] * 0.5, query_dir[2] * 0.5];

        let nearest_shards = registry.nearest_shards(&query_pos, 6);
        println!(
            "Nearest shards: {}",
            nearest_shards
                .iter()
                .map(|(a, d)| format!("{} ({:.3})", a.shard_id, d))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let mut loaded_shards = Vec::new();
        for (anchor, _) in &nearest_shards {
            let path = km_dir.join(format!("{}.km", anchor.shard_id));
            if let Ok(shard) = read_shard(&path) {
                loaded_shards.push(shard);
            }
        }

        // Full, untruncated description here -- this is what actually goes
        // into the RAG prompt below. Truncating at this point (the old
        // `.chars().take(100)`) meant the LLM never saw anything past the
        // 100th character of a concept's description, silently cutting off
        // facts (a dollar figure, a date) that happened to land later in
        // the sentence -- indistinguishable from the concept just not
        // having that information, except the model would then confabulate
        // something plausible-looking instead of saying so.
        let mut ranked: Vec<(f64, String, String, String)> = Vec::new();
        for shard in &loaded_shards {
            for (_, concept) in &shard.concepts {
                if let ManifoldCoord::Hyperbolic { position } = &concept.coordinate {
                    let dist = poincare_distance(&query_pos, position);
                    ranked.push((
                        dist,
                        concept.raw_term.clone(),
                        concept.label.clone(),
                        concept.description.clone(),
                    ));
                }
            }
        }

        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Truncation lives here now -- print-only, never touches what the
        // model receives.
        println!("Top 5 nearest concepts:");
        for (dist, raw_term, label, desc) in ranked.iter().take(5) {
            let preview: String = desc.chars().take(100).collect();
            let ellipsis = if desc.chars().count() > 100 { "..." } else { "" };
            println!("  [{dist:.4}] {raw_term} -> \"{label}\"");
            println!("           {preview}{ellipsis}");
        }
        println!();

        // Generation
        let enrich_config = EnrichConfig::default();
        let prompt = build_rag_prompt(query, &ranked.iter().take(5).cloned().collect::<Vec<_>>());
        match chat(&prompt, &enrich_config.summarize_model, &enrich_config) {
            Ok(answer) => println!("\nAnswer:\n{answer}\n"),
            Err(e) => println!("\nGeneration failed: {e}\n"),
        }
    }
}