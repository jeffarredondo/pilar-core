use std::path::Path;
use crate::types::Chunk;
use serde::Deserialize;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct IngestConfig {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            chunk_size: 2000,
            overlap: 200,
        }
    }
}

// ── Chunking ──────────────────────────────────────────────────────────────────

/// Sliding-window chunker — direct port of the Python prototype's
/// chunk_text. Character-indexed deliberately, not byte-indexed: Python's
/// text[start:end] slices by Unicode character, and naive byte-slicing
/// in Rust would panic or silently produce different chunk boundaries on
/// any non-ASCII text (smart quotes, em-dashes — both realistic in
/// Gutenberg-era texts and SEC filings). Collecting into Vec<char> costs
/// extra memory but keeps the character-vs-byte distinction impossible
/// to get wrong by accident.
///
/// One real addition over Python: each chunk carries source_path and an
/// approximate starting source_line. Python's version flattened every
/// source into one untracked list of strings, losing exactly the
/// traceability the rest of this pipeline (raw_term, Concept.source_path)
/// was built to preserve — this isn't new behavior, it's continuing a
/// decision already made elsewhere in this codebase.
pub fn chunk_text(text: &str, source_path: &Path, config: &IngestConfig) -> Vec<Chunk> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    if n == 0 {
        return vec![];
    }

    debug_assert!(
        config.overlap < config.chunk_size,
        "overlap must be smaller than chunk_size or the sliding window can't advance"
    );

    // Precomputed once, in character-index space, so each chunk's
    // starting line is a binary search instead of a fresh linear scan
    // from the start of the text — O(n log n) total instead of
    // O(chunks * n), which matters once a source gets into the
    // hundred-chunk range.
    let newline_positions: Vec<usize> = chars
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c == '\n')
        .map(|(i, _)| i)
        .collect();

    // Floored at 1 so a misconfigured overlap >= chunk_size can't hang
    // forever even if the debug_assert above is compiled out in
    // release. Python's version has no equivalent guard and would loop
    // indefinitely on the same bad config -- not a behavior worth
    // preserving, just a latent bug it happened to never hit.
    let stride = config.chunk_size.saturating_sub(config.overlap).max(1);

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < n {
        let end = (start + config.chunk_size).min(n);
        let chunk_text: String = chars[start..end].iter().collect();

        // 1-indexed: how many newlines occurred strictly before this
        // chunk's start. The newline terminating a line is treated as
        // still belonging to that line, not the next one -- consistent
        // with how most editors report line numbers.
        let line = newline_positions.partition_point(|&p| p < start) + 1;

        chunks.push(Chunk {
            text: chunk_text,
            source_path: source_path.to_path_buf(),
            source_line: Some(line),
        });

        start += stride;
    }

    chunks
}

/// Thin I/O wrapper around chunk_text -- reads a file, then chunks it.
/// Kept separate so chunk_text itself stays pure and testable without
/// touching disk, same split as embed.rs's embed() vs parse_embedding().
pub fn chunk_file(path: &Path, config: &IngestConfig) -> std::io::Result<Vec<Chunk>> {
    let text = std::fs::read_to_string(path)?;
    Ok(chunk_text(&text, path, config))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_empty_text() {
        let chunks = chunk_text("", &PathBuf::from("empty.txt"), &IngestConfig::default());
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_short_text_single_chunk() {
        let config = IngestConfig::default();
        let chunks = chunk_text("a short piece of text", &PathBuf::from("a.txt"), &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "a short piece of text");
    }

    #[test]
    fn test_source_path_propagates_to_every_chunk() {
        let config = IngestConfig {
            chunk_size: 5,
            overlap: 1,
        };
        let path = PathBuf::from("spacex_s1.txt");
        let chunks = chunk_text("a longer piece of text than one chunk", &path, &config);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.source_path == path));
    }

    #[test]
    fn test_sliding_window_overlap() {
        // chunk_size=10, overlap=2 -> stride=8. First chunk is chars
        // 0..10, second is chars 8..18 -- chars 8 and 9 should appear
        // in both.
        let config = IngestConfig {
            chunk_size: 10,
            overlap: 2,
        };
        let text = "0123456789ABCDEFGHIJ";
        let chunks = chunk_text(text, &PathBuf::from("t.txt"), &config);

        assert_eq!(chunks[0].text, "0123456789");
        assert_eq!(chunks[1].text, "89ABCDEFGH");
    }

    #[test]
    fn test_multibyte_utf8_does_not_panic_or_corrupt() {
        // Curly quotes and an em-dash -- realistic in both Gutenberg
        // texts and modern filings, and exactly what byte-indexed
        // slicing would mishandle.
        let text = "\u{201c}Extraordinary Popular Delusions\u{201d} \u{2014} a study in mania.";
        let config = IngestConfig {
            chunk_size: 10,
            overlap: 2,
        };
        let chunks = chunk_text(text, &PathBuf::from("mackay.txt"), &config);

        // Reassembling via the same stride should reproduce recognizable
        // text -- the real assertion is just that this didn't panic and
        // each chunk is exactly chunk_size characters (not corrupted
        // byte-boundary fragments).
        assert!(!chunks.is_empty());
        for c in &chunks[..chunks.len() - 1] {
            assert_eq!(c.text.chars().count(), 10);
        }
    }

    #[test]
    fn test_source_line_tracks_position_correctly() {
        // 6 chunks at stride 8 over known newline positions -- see the
        // worked-through math in the PR/commit message; verified by
        // hand against this exact text before writing the assertions.
        let text = "line one\nline two\nline three\nline four\nline five";
        let config = IngestConfig {
            chunk_size: 10,
            overlap: 2,
        };
        let chunks = chunk_text(text, &PathBuf::from("lines.txt"), &config);

        assert_eq!(chunks.len(), 6);
        assert_eq!(chunks[0].source_line, Some(1)); // start=0
        assert_eq!(chunks[2].source_line, Some(2)); // start=16, in "line two"
        assert_eq!(chunks[3].source_line, Some(3)); // start=24, in "line three"
        assert_eq!(chunks[4].source_line, Some(4)); // start=32, in "line four"
        assert_eq!(chunks[5].source_line, Some(5)); // start=40, in "line five"
    }

    #[test]
    #[should_panic]
    fn test_overlap_greater_than_chunk_size_trips_debug_assert() {
        let config = IngestConfig {
            chunk_size: 5,
            overlap: 10,
        };
        chunk_text("some text long enough to matter", &PathBuf::from("t.txt"), &config);
    }
}