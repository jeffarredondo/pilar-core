use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pilar_core::pipeline::PipelineConfig;
use pilar_core::km::{read_registry, read_shard};
use pilar_core::placement::{Projections, PlacementConfig};
use pilar_core::embed::{embed, EmbedConfig};
use pilar_core::geometry::poincare_distance;
use pilar_core::types::{Concept, ManifoldCoord};

fn main() {
    let km_dir = PathBuf::from("./km_output");
    let queries = vec![
        "What is SpaceX's revenue and launch strategy?",
        "Tell me about witchcraft trials in Europe",
        "Wall Street speculation and stock market panic",
        "philosopher's stone and alchemy",
    ];

    // Load registry
    let registry_path = km_dir.join("registry.km");
    let registry = read_registry(&registry_path).expect("failed to read registry");
    println!("Registry loaded: {} shards", registry.anchor_count());
    println!();

    let embed_config = EmbedConfig::default();

    for query in &queries {
        println!("Query: {query}");
        println!("{}", "─".repeat(60));

        // Embed the query
        let query_embedding = embed(query, &embed_config).expect("embed failed");

        // Project to manifold coordinate using same seed as ingest
        let projections = Projections::new(query_embedding.len(), PlacementConfig::default().projection_seed);
        let dir = projections.hyperbolic_direction(&query_embedding);
        // Use strength=0.5 as neutral query position
        let r = 1.0 - 0.9 * 0.5_f64;
        let query_position = [dir[0] * r, dir[1] * r, dir[2] * r];

        // Find nearest shards via registry
        let nearest_shards = registry.nearest_shards(&query_position, 3);
        println!("Nearest shards: {}", nearest_shards.iter().map(|(a, d)| format!("{} ({:.3})", a.shard_id, d)).collect::<Vec<_>>().join(", "));

        // Load those shards and collect all concepts
        let mut candidates: Vec<(f64, &'static str, String, String)> = Vec::new();
        let mut loaded_shards = Vec::new();

        for (anchor, _) in &nearest_shards {
            let path = km_dir.join(format!("{}.km", anchor.shard_id));
            if let Ok(shard) = read_shard(&path) {
                loaded_shards.push(shard);
            }
        }

        // Compute distances and rank
        let mut ranked: Vec<(f64, String, String, String)> = Vec::new();
        for shard in &loaded_shards {
            for (_, concept) in &shard.concepts {
                if let ManifoldCoord::Hyperbolic { position } = &concept.coordinate {
                    let dist = poincare_distance(&query_position, position);
                    ranked.push((dist, concept.raw_term.clone(), concept.label.clone(), concept.description.chars().take(100).collect()));
                }
            }
        }

        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        println!("Top 5 nearest concepts:");
        for (dist, raw_term, label, desc) in ranked.iter().take(5) {
            println!("  [{dist:.4}] {raw_term} -> \"{label}\"");
            println!("           {desc}...");
        }
        println!();
    }
}