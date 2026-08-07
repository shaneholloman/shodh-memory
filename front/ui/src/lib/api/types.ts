/**
 * Wire types, each mirroring a Rust struct that was read before it was typed.
 * The citation on each one is load-bearing: if a field is not listed in the
 * handler it is not listed here, and nothing is added speculatively.
 */

/** `RecallExperience` — src/handlers/types.rs:256 */
export interface RecallExperience {
  content: string;
  memory_type: string | null;
  tags: string[];
  /** `[lat, lon, alt]` WGS84. Absent unless the memory carries coordinates —
   *  nothing in the Rust path derives them, they are caller-supplied only. */
  geo_location?: [number, number, number];
}

/**
 * `ScoreAttribution` — src/memory/types.rs.
 *
 * Present only when the request set `debug: true`. There is no endpoint to
 * fetch it for one memory afterwards; it is all-or-nothing per query.
 */
export interface ScoreAttribution {
  memory_id: string;
  rrf_base: number;
  graph_rrf: number;
  hybrid_rrf: number;
  hebbian_boost: number;
  attribute_boost: number;
  temporal_prefilter_boost: number;
  temporal_fact_boost: number;
  interference_adjustment: number;
  prospective_boost: number;
  fact_source_boost: number;
  ontological_boost: number;
  importance_factor: number;
  recency_factor: number;
  arousal_factor: number;
  credibility_factor: number;
  feedback_multiplier: number;
  quality_gate: number;
  final_score: number;
  /** Which retrieval legs contributed this memory. */
  sources: string[];
}

/** `RecallMemory` — src/handlers/types.rs:243 */
export interface RecallMemory {
  id: string;
  experience: RecallExperience;
  importance: number;
  created_at: string;
  score: number;
  tier: string;
  score_attribution?: ScoreAttribution;
}

/** `RecallFact` — src/handlers/types.rs */
export interface RecallFact {
  id: string;
  fact: string;
  confidence: number;
  support_count: number;
  related_entities: string[];
}

/** `RecallTodo` — src/handlers/types.rs */
export interface RecallTodo {
  id: string;
  short_id: string;
  content: string;
  status: string;
  priority: string;
  project: string | null;
  due_date: string | null;
  score: number;
}

/** `RecallLineageEdge` — src/handlers/types.rs. Causal edges between recalled
 *  memories: this is the trust chain Chain 2 walks. */
export interface RecallLineageEdge {
  from: string;
  to: string;
  relation: string;
  confidence: number;
}

/**
 * `RecallResponse` — src/handlers/types.rs:164.
 *
 * The collection fields carry `skip_serializing_if = "Vec::is_empty"` on the
 * Rust side, so they are genuinely absent rather than empty when there is
 * nothing to send. Every one is optional here for that reason.
 */
export interface RecallResponse {
  memories: RecallMemory[];
  count: number;
  todos?: RecallTodo[];
  todo_count?: number;
  facts?: RecallFact[];
  fact_count?: number;
  lineage?: RecallLineageEdge[];
  lineage_count?: number;
}

/** `RecallRequest` — src/handlers/types.rs. Only the fields this UI sends. */
export interface RecallRequest {
  user_id: string;
  query: string;
  limit?: number;
  mode?: "hybrid" | "semantic" | "associative" | "temporal" | "spatial";
  /** Produces `score_attribution` on every returned memory. */
  debug?: boolean;
  offset?: number;
}
