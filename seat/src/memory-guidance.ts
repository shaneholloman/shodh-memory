/**
 * Memory-use guidance appended to the seat's base system prompt.
 *
 * Every line here targets a failure measured in the baseline evaluation
 * (seat/eval/memory-guidance-ab.mjs, baseline phase, claude-haiku-4-5, N=3):
 *
 * - recall_memory was called ZERO times across 63 baseline cases; the model
 *   treated the 3-memory auto-surfaced block as the whole of memory and
 *   declared details "not recorded" without ever searching. → bullets 1, 2, 4.
 * - Multi-evidence chain cases failed by citing one link (forward-trace 0/3,
 *   propulsion-chain 1/3). → bullet 3, graph-tools bullet.
 * - The absence case drew fabricated citations 3/3. → citation bullet.
 * - MCP-sourced full-UUID citations appeared 4 times; the seat's citation
 *   contract ([mem:<8 hex>], src/feedback.ts) cannot parse them, and MCP
 *   recall results bypass the reinforcement loop (only native recall passes
 *   through onSurfaced, src/memory-tools.ts). → tool-choice + id bullets.
 * - The correction-pair case cited the stale report over the correction
 *   (casualties-identity 1/3). → last bullet.
 *
 * Write hygiene was 6/6 at baseline — deliberately NOT addressed here; the
 * base prompt's existing bullets already cover it and every added line costs
 * context on every turn.
 *
 * The A/B harness injects this text verbatim as the treatment arm, so what is
 * measured is byte-identical to what would ship.
 *
 * ── A/B VERDICT (claude-haiku-4-5, 8 repeats/arm, 168 trials/arm, interleaved,
 *    fresh seeded user per run, paired permutation test) ──────────────────────
 *
 *   control 60.7% ± 9.4pp · guidance 67.9% ± 6.1pp · +7.1pp, p = 0.167
 *   needle stratum (retrieval-only evidence): 52.1% → 68.8% (z = 1.67)
 *   native recall_memory usage: 0.02 → 0.35 calls/case (citations sourced
 *   from native recall: 3 → 78); full-UUID citation-contract violations 1 → 0
 *   REGRESSION (nuanced): the absence case fell 2/8 → 0/8 under the suite's
 *   strict zero-citation rule, but the arm-B transcripts all answer honestly
 *   ("no records of rail strikes…") while citing REAL adjacent memories as
 *   context — over-citation, not fabrication. The fabricated-citations
 *   counter (7 → 14) is dominated in BOTH arms by the model echoing the id
 *   of a memory it has just written (write-capture), which the corpus-
 *   membership check mislabels; genuinely unexplained ids: ~2 vs ~8.
 *
 * NOT WIRED into BASE_SYSTEM_PROMPT: the pre-registered bar was a measured
 * improvement, and p = 0.167 does not clear it. The effect is directionally
 * positive and the mechanism (the model actually searching memory instead of
 * treating the 3-memory proactive block as the whole store) demonstrably
 * moved, but establishing a ~7pp effect at this variance needs roughly 30+
 * repeats per arm. Anyone revisiting: sharpen the citation-restraint line
 * (or the absence rule) against contextual over-citation, then re-run
 * seat/eval/memory-guidance-ab.mjs before wiring this in.
 */
export const MEMORY_GUIDANCE = `Working memory effectively:
- The auto-surfaced memory block is a starting point, not the whole of memory. Never say something is absent from memory until recall_memory has searched for it and come back empty.
- Retrieve with recall_memory. Build queries from concrete entity names, identifiers, and quantities rather than abstractions; if a cue misses, reformulate with alternative concrete terms (names, short forms, codes) before giving up.
- Answer multi-part or causal-chain questions one link at a time: recall each link, then answer citing every memory used. A single recall rarely covers a chain.
- When a recalled memory names an entity you need more detail on, recall again with that entity's exact name.
- If graph tools are connected (trace_lineage, explore_entity, list_entities, list_anomalies, search_facts), use them for structure: trace_lineage for what led to or resulted from a memory (causal edges are typed FROM→TO: Caused, ResolvedBy, InformedBy, TriggeredBy, SupersededBy), explore_entity to walk a named person, place, or system, list_anomalies for "anything unusual?" questions. For plain retrieval always use recall_memory, not an MCP recall variant — only native recall feeds the learning loop.
- Cite [mem:<id>] with the 8-character id shown in recall results; for a long UUID, use its first 8 characters. Cite only memories that directly support the sentence they follow; when memory has nothing on a topic, say so plainly and cite nothing.
- Memories can correct earlier ones. When recalled memories conflict, the correction or newer memory wins; cite it and note what it superseded.`;
