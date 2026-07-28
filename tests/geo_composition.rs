//! Geo composition test (Task 2 of the geotemporal plan).
//!
//! Composes `Query::geo_filter` with semantic retrieval: an in-radius memory
//! that is semantically UNRELATED to the query text must still surface via
//! the Layer 0.45 geo prefetch + additive pool injection, while the hard geo
//! predicate (`Query::matches`) at hydration must still exclude out-of-radius
//! and geo-less memories.
//!
//! A 3-memory corpus cannot discriminate this: `vector_top_k = query.max_results
//! * 3` (30 by default) trivially returns every memory in a tiny corpus
//! regardless of relevance, so the semantically-unrelated in-radius memory
//! rides in via the ordinary vector leg and the injection assertion passes
//! even with no engine change. `FILLER_COUNT` fillers crowd the natural
//! candidate window so only the Layer 0.45/4.46 injection path can surface it
//! — see the `FILLER_COUNT` doc comment below for the full argument.
//!
//! Storage lives under `std::env::temp_dir()` via `tempfile::TempDir`, NOT
//! under any OneDrive-watched path — the tantivy file-watcher corrupts
//! commits under watched directories (see the BM25 onedrive finding).

use shodh_memory::memory::storage::SearchCriteria;
use shodh_memory::memory::{
    Experience, ExperienceType, GeoFilter, MemoryConfig, MemorySystem, Query, RetrievalMode,
};
use tempfile::TempDir;

/// Regression test for the storage-layer geo index in isolation from the
/// rest of the retrieval pipeline: a memory's geohash index entry is written
/// synchronously as part of `remember()` (storage.rs `store_inner` calls
/// `update_indices` before returning), so `advanced_search(ByLocation)` must
/// find it immediately, with no consolidation/promotion delay. Layer 0.45's
/// prefetch depends on this holding — if this test ever regresses, the bug
/// is in storage/indexing, not in the Layer 0.45/4.46 wiring in
/// `semantic_retrieve_inner`.
#[test]
fn advanced_search_finds_geo_memory_synchronously() {
    let (system, _temp) = setup_memory_system();
    let inside = [39.2904, -76.6122, 0.0];
    store_with_geo(&system, "harbor patrol completed", Some(inside));

    let found = system
        .advanced_search(SearchCriteria::ByLocation {
            lat: 39.2904,
            lon: -76.6122,
            radius_meters: 25_000.0,
        })
        .expect("advanced_search should succeed");

    let texts: Vec<&str> = found.iter().map(|m| m.experience.content.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.contains("harbor patrol")),
        "advanced_search(ByLocation) did not find the geo-tagged memory: {texts:?}"
    );
}

/// Create test memory system with storage under `std::env::temp_dir()`.
fn setup_memory_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 200,
        auto_compress: false,
        compression_age_days: 30,
        importance_threshold: 0.7,
    };

    let memory_system = MemorySystem::new(config, None).expect("Failed to create memory system");
    (memory_system, temp_dir)
}

/// Store a memory with optional geo_location `[lat, lon, alt]`.
fn store_with_geo(system: &MemorySystem, content: &str, geo: Option<[f64; 3]>) {
    let experience = Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        geo_location: geo,
        ..Default::default()
    };
    system
        .remember(experience, None)
        .expect("remember should succeed");
}

/// Number of filler memories needed to push A (semantically unrelated to the
/// query) outside the natural vector-search candidate window.
///
/// `semantic_retrieve_inner`'s Layer 3 sets `vector_top_k = query.max_results
/// * 3` (mod.rs, grep "let vector_top_k ="); with `DEFAULT_MAX_RESULTS = 10`
/// that window is 30. In a bare 3-memory corpus, Vamana's kNN trivially
/// returns all 3 regardless of relevance (k > corpus size), so A enters the
/// fused pool via the ordinary vector leg and the injection assertion below
/// passes vacuously — verified empirically: the 3-memory version of this
/// test is GREEN even without the Layer 0.45/4.46 engine change. These
/// fillers are moderately relevant to the query (share one of
/// "bridge"/"inspection" but never the exact two-word phrase) so they rank
/// between B/C (exact-phrase match, stay in-pool so the real hydration
/// predicate is what excludes them) and A (zero lexical/semantic overlap,
/// pushed below rank 30). None contain "report" or "summary" — those
/// substrings are asserted absent from the final results.
const FILLER_COUNT: usize = 40;

// Three memories: (A) geo-tagged inside radius, semantically UNRELATED to query text;
// (B) geo-tagged outside radius, semantically related; (C) no geo, semantically related.
// Plus FILLER_COUNT geo-less, moderately-relevant filler memories so A cannot
// ride into the fused pool via the ordinary vector/BM25 legs (see FILLER_COUNT doc).
// Query: Hybrid mode, text matching B/C, geo_filter centered on A's location.
#[test]
fn hybrid_mode_geo_filter_composes() {
    let (system, _temp) = setup_memory_system();

    let inside = [39.2904, -76.6122, 0.0]; // Baltimore
    let outside = [51.5074, -0.1278, 0.0]; // London
    store_with_geo(&system, "harbor patrol completed", Some(inside)); // A
    store_with_geo(&system, "bridge inspection report", Some(outside)); // B
    store_with_geo(&system, "bridge inspection summary", None); // C

    for i in 0..FILLER_COUNT {
        let content = if i % 2 == 0 {
            format!(
                "traffic diverted while crews repaired the old stone bridge near the river, case {i}"
            )
        } else {
            format!("quarterly safety inspection completed for the downtown facility, case {i}")
        };
        store_with_geo(&system, &content, None);
    }

    let query = Query::builder()
        .query_text("bridge inspection")
        .retrieval_mode(RetrievalMode::Hybrid)
        .geo_filter(GeoFilter::new(39.2904, -76.6122, 25_000.0))
        .build();
    let results = system
        .recall_with_diagnostics(&query)
        .expect("recall should succeed");

    let texts: Vec<&str> = results
        .memories
        .iter()
        .map(|m| m.experience.content.as_str())
        .collect();
    // Hard predicate: B (outside radius) and C (no geo) MUST be excluded.
    assert!(
        !texts.iter().any(|t| t.contains("report")),
        "outside-radius memory leaked: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.contains("summary")),
        "geo-less memory leaked through geo filter: {texts:?}"
    );
    // Recall via injection: A is semantically unrelated to the query — it can only
    // arrive via the geo prefetch union. This is the recall-improvement assertion.
    assert!(
        texts.iter().any(|t| t.contains("harbor patrol")),
        "geo prefetch did not inject in-radius memory: {texts:?}"
    );
}
