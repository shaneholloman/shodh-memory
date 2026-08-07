#!/usr/bin/env node
/**
 * Seed the demonstration corpus against a RUNNING backend.
 *
 *   SHODH_API_URL=http://127.0.0.1:3098 SHODH_API_KEY=... USER_ID=demo \
 *     node seat/eval/seed-demo-corpus.mjs
 *
 * The corpus is written to exercise every subsystem the workbench renders,
 * deliberately:
 *  - explicit causal language ("because", "led to", "triggered") for the
 *    causal extractors → typed lineage edges, not bare co-occurrence;
 *  - named people, organisations, places and systems → a visibly typed
 *    ontology once NER typing lands (person/org/location/technology);
 *  - place names in text → toponym linking (gazetteer), distinct from…
 *  - …real recorded coordinates on field events → the geo layer;
 *  - repeated entities across memories → canonicalization + Hebbian edges;
 *  - a deliberate factual correction pair → the facts contradiction path;
 *  - open todos → Tasks.
 *
 * Idempotence: re-running adds duplicates. Seed fresh stores only.
 */
const BASE = (process.env.SHODH_API_URL ?? "http://127.0.0.1:3098").replace(/\/+$/, "");
const KEY = process.env.SHODH_API_KEY ?? process.env.SHODH_API_KEYS?.split(",")[0]?.trim();
const USER = process.env.USER_ID ?? "demo";
if (!KEY) { console.error("SHODH_API_KEY required"); process.exit(2); }

const H = { "X-API-Key": KEY, "Content-Type": "application/json" };
async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, { method: "POST", headers: H, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json();
}

/** [content, type, geo?] — geo only where the event has a real location. */
const MEMORIES = [
  // Causal chain 1 — the incident
  ["Container ship Dali lost propulsion at 01:24 local time because an electrical breaker tripped during departure from the Port of Baltimore.", "observation", [39.264, -76.6015, 0]],
  ["The loss of propulsion led to the Dali drifting off the channel heading despite the crew dropping anchor.", "observation", [39.24, -76.575, 0]],
  ["The drifting vessel struck a support pier of the Francis Scott Key Bridge, which triggered the collapse of the main truss spans into the Patapsco River.", "observation", [39.217, -76.5285, 0]],
  ["Because the wreckage blocked the shipping channel, the Port of Baltimore suspended all vessel traffic.", "decision", [39.2604, -76.6122, 0]],
  // Causal chain 2 — consequences
  ["The port suspension halted roll-on/roll-off automobile shipments, so Maersk rerouted its services to the Port of New York and New Jersey.", "decision", [40.684, -74.145, 0]],
  ["Overflow automobile volume was diverted south to the Port of Virginia at Norfolk as a result of the closure.", "decision", [36.9312, -76.321, 0]],
  ["Coal exports from the CSX Curtis Bay terminal stopped for six weeks because bulk carriers could not transit the blocked channel.", "observation", [39.2205, -76.5866, 0]],
  // Investigation — people and orgs for the ontology
  ["NTSB chair Jennifer Homendy stated the investigation would examine the Dali's electrical system, after inspectors recovered the voyage data recorder.", "observation", null],
  ["Synergy Marine Group, the ship manager, confirmed the crew reported electrical failures during port inspections in the days before departure.", "observation", null],
  ["Captain Maynard of the pilots association credited the pilot's mayday call with stopping bridge traffic, which prevented further casualties.", "learning", null],
  // Technical layer
  ["The bridge lacked pier protection dolphins that current design standards require, a factor engineers said contributed to the total collapse.", "learning", null],
  ["Salvage crews used the floating crane Chesapeake 1000 to cut the collapsed truss sections for channel clearance.", "task", [39.217, -76.5285, 0]],
  // Deliberate correction pair — exercises the facts contradiction path
  ["Initial reports said four crew members were injured in the collapse.", "observation", null],
  ["Corrected report: no crew members aboard the Dali were injured; the casualties were road workers on the bridge deck.", "observation", null],
  // Unrelated island — proves the graph renders disconnection honestly
  ["The Vamana index rebuild cut recall latency from 340ms to 92ms on the evaluation corpus.", "learning", null],
];

const TODOS = [
  { content: "Review insurance exposure for charter agreements affected by the port closure", priority: "urgent" },
  { content: "Verify rerouted cargo volumes at NY/NJ against manifest data", priority: "high" },
  { content: "Compile channel clearance timeline from salvage reports", priority: "medium", due_date: "2026-08-12" },
];

let ok = 0;
for (const [content, memory_type, geo] of MEMORIES) {
  await post("/api/remember", {
    user_id: USER, content, memory_type,
    ...(geo ? { geo_location: geo } : {}),
  });
  ok += 1;
  process.stdout.write(`\rmemories: ${ok}/${MEMORIES.length}`);
}
console.log();
for (const todo of TODOS) await post("/api/todos/add", { user_id: USER, ...todo });
console.log(`todos: ${TODOS.length}`);

const recall = await post("/api/recall", { user_id: USER, query: "why did the bridge collapse", limit: 8, debug: true });
console.log(`recall smoke: ${recall.count} memories, ${recall.lineage?.length ?? 0} lineage edges, ${recall.facts?.length ?? 0} facts`);
console.log("seeded.");
