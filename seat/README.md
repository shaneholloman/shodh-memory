# seat — conversation harness

Server-side agent harness for the shodh-memory conversation seat. It owns the
agent loop, streams structured events to a client, and closes two learning
loops against the Rust backend on every turn. Memory operations are never
opaque: every recall carries ids, scores, and full per-memory
`ScoreAttribution`; every learning update is a reviewable, revertible ledger
event.

Built on [`pi`](https://github.com/earendil-works/pi) (`@earendil-works/pi-agent-core`
for the agent loop, `@earendil-works/pi-ai` for the unified multi-provider LLM
API). Provider credentials resolve from environment variables inside this
process and never reach a client.

## Architecture

```
client (SSE) ──► SeatServer (node:http)
                    │
                    ├─ ModelRegistry ── pi builtin providers (env-key auth)
                    │                └─ local providers: Ollama / LM Studio
                    │                   (custom pi provider, openai-completions
                    │                    API + per-model baseUrl)
                    │
                    ├─ Conversation ── pi Agent (loop, tools, streaming)
                    │     ├─ memory tools (native HTTP → Rust backend)
                    │     ├─ MCP bridge tools (stdio, any MCP server)
                    │     └─ learning loops (see below)
                    │
                    ├─ ShodhBackend ── HTTP client for the Rust API
                    │                  (X-API-Key, shapes transcribed from
                    │                   src/handlers/*)
                    └─ LearningLedger ── append-only JSONL, revert support
```

## Memory as a first-class tool

`recall_memory` calls `POST /api/recall` with `debug: true`, so every result
carries the backend's full score attribution (RRF base, graph/hybrid RRF,
hebbian boost, importance/recency/arousal/credibility factors,
`feedback_multiplier`, quality gate, final score, contributing sources). The
tool returns a compact text rendering to the model and emits a
`memory_recall` SSE event with the complete structured payload — retrieved
ids, scores, attribution, related facts, todos, and lineage edges — for the UI
to render as its own element.

The system prompt asks the model to cite memories inline as `[mem:<id>]`;
citations feed the reinforcement loop below.

## Two learning loops, one substrate

Both loops store and strengthen state through the same shodh-memory machinery.
There is no second state store.

**Loop 1 — memory-level (user scope), two legs with strict ownership.**
`/api/reinforce` moves importance, Hebbian associations, and entity salience —
but **not** feedback momentum. The only backend path that writes momentum (the
`feedback_multiplier` in score attribution) is `POST /api/proactive_context`
(`src/handlers/recall.rs:1310-1720`). The seat drives both:

- *Implicit/momentum leg* — every turn calls `proactive_context` with the new
  user message as `context`, the previous assistant message as
  `previous_response`, the new user message as `user_followup`, and the
  previous run's tool actions (`ToolAction` shape from
  `src/memory/feedback.rs`; native memory tools excluded — a recall cue
  trivially overlaps memory content and would fake a usage signal). The
  backend evaluates the previous turn's pending surfaced set (momentum,
  context fingerprints, temporal credits) and applies its own
  `reinforce_recall` + Hebbian pass for helpful/misleading classifications.
  The memories it surfaces are all injected into the system prompt (surfaced
  set == seen set, otherwise the implicit loop penalizes memories the model
  never saw) and are **owned by this leg**. The `proactive_context` SSE event
  exposes the surfaced set and the feedback outcome (reinforced/weakened ids).
- *Explicit leg* — memories recalled by the `recall_memory` tool and **not**
  proactive-surfaced that turn are reinforced through `POST /api/reinforce`:
  - *helpful* — cited (`[mem:id]`) or token overlap ≥ 0.1 (mirrors
    `calculate_entity_overlap` / `OVERLAP_WEAK_THRESHOLD`);
  - *neutral* — surfaced but unused;
  - *misleading* — negative follow-up keywords (verbatim `NEGATIVE_KEYWORDS`
    list), applied to the previous turn's surfaced set minus proactive-owned
    ids.

The id-level ownership split is what prevents double-counting: a memory
surfaced by both channels in one turn is reinforced exactly once (by the
implicit leg). Known seam: tool-recalled-only memories get no momentum, because
the backend ties momentum to its own pending-set lifecycle
(`set_pending`/`take_pending`, single slot per `user_id`) — moving them would
require a backend change, not a harness workaround. For the same reason the
seat guards concurrent proactive calls per `user_id` in-process (feedback
fields are skipped when a call is in flight, mirroring `mcp-server/index.ts`);
a second process using the same `user_id` cannot be guarded from here.

`auto_ingest` is explicitly `false` on every call: the backend default (true)
silently ingests the previous response as memories, which would bypass the
ledger. Seat writes stay deliberate and ledgered.

**Loop 2 — harness-level (harness scope).** The harness gets better at its own
job by storing operational lessons *as memories* in an isolated namespace,
`<user_id>.seat-harness`. Scope isolation uses the backend's existing
per-`user_id` seam: memory store, knowledge graph, and feedback state are all
keyed by `user_id` directory (`src/handlers/state.rs`), so the two scopes can
never share retrieval, Hebbian co-activation, or feedback statistics.

Harness learnings enter the scope three ways:

- automatic capture of empty recalls (cue recorded with rephrasing advice);
- automatic capture of tool failures;
- the model's own `record_seat_learning` tool (operational lessons only).

Before each turn the harness recalls from its scope with the user message as
cue and injects strong matches (score ≥ 0.25, max 3) as a labeled
system-prompt block for that turn only, emitting `harness_learning_applied`.
Injected learnings are reinforced by the same rules as user memories — the
loop is closed in both scopes.

## Learning ledger

Every update either loop makes is appended to a JSONL ledger
(`<data-dir>/learning-ledger.jsonl`) *before* the conversation moves on:
memory writes, every reinforcement (with per-memory overlap values and
trigger), and reverts. Reverts are appended events referencing the original —
nothing is mutated.

Revert semantics are honest about what the backend supports:

- memory writes revert exactly (`DELETE /api/memory/{id}`);
- helpful/misleading reinforcements revert by a *compensating* opposite
  outcome through the same `/api/reinforce` path — the backend's EMA-with-
  inertia momentum update is not exactly invertible, and the revert event says
  so;
- neutral reinforcements record access only; nothing to compensate.

## HTTP API

| Method | Path | Body / notes |
|---|---|---|
| GET | `/healthz` | seat + backend health (unauthenticated) |
| GET | `/v1/models?refresh=1` | available models; `refresh` re-probes local endpoints |
| POST | `/v1/conversations` | `{user_id, provider, model, system_prompt?}` |
| GET | `/v1/conversations/{id}` | state + transcript |
| POST | `/v1/conversations/{id}/messages` | `{text}` → SSE stream of events |
| PATCH | `/v1/conversations/{id}/model` | `{provider, model}` — swap model mid-conversation; transcript and retrieved evidence unchanged |
| GET | `/v1/learning/events?limit&conversation_id` | ledger review |
| POST | `/v1/learning/revert` | `{event_id}` |

SSE event types: `turn_start`, `text_delta`, `thinking_delta`,
`tool_call_start`, `tool_call_end`, `memory_recall`, `proactive_context`,
`memory_write`, `memory_reinforce`, `harness_learning_applied`,
`model_changed`, `usage`
(per-call token counts and cost from pi's `Usage`), `turn_end`, `agent_end`,
`error`. Payloads are defined in `src/events.ts`.

## Local models

pi ships no Ollama/LM Studio/vLLM provider, but every pi `Model` carries its
own `baseUrl` and the `openai-completions` implementation dials it directly.
The registry therefore registers two custom providers (`ollama`, `lmstudio`)
built with `createProvider` + `openAICompletionsApi()`, keyless auth, and
dynamic model discovery via `GET {baseUrl}/models`. Any other OpenAI-compatible
endpoint works the same way.

## MCP bridge

pi has no MCP client (its README: "No MCP"), so `src/mcp.ts` bridges MCP
servers into the agent loop: stdio transport, `listTools` → pi `AgentTool`s
named `mcp__<server>__<tool>`, plain JSON Schema passed through (pi's
validator handles non-TypeBox schemas). Configure with
`SEAT_MCP_SERVERS=/path/to/servers.json`:

```json
{ "servers": [ { "name": "shodh-memory", "command": "node", "args": ["mcp-server/dist/index.js"] } ] }
```

The native memory tools remain the recall/remember path of record — they carry
attribution that MCP text framing cannot.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SHODH_API_URL` | `http://127.0.0.1:3030` | Rust backend (or `SHODH_HOST`/`SHODH_PORT`) |
| `SHODH_API_KEY` | — (required) | backend `X-API-Key` (falls back to `SHODH_DEV_API_KEY`, `SHODH_API_KEYS`) |
| `SEAT_HOST` / `SEAT_PORT` | `127.0.0.1` / `3141` | bind address |
| `SEAT_AUTH_TOKEN` | — | bearer token; mandatory for non-loopback binds |
| `SEAT_DATA_DIR` | `%LOCALAPPDATA%\shodh\seat-harness` (win) / XDG data dir | ledger location — deliberately outside watched/synced folders |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434/v1` | Ollama OpenAI-compat endpoint |
| `LMSTUDIO_BASE_URL` | `http://127.0.0.1:1234/v1` | LM Studio endpoint |
| `SEAT_LOCAL_CONTEXT_WINDOW` / `SEAT_LOCAL_MAX_TOKENS` | `32768` / `8192` | advertised limits for local models |
| `SEAT_MCP_SERVERS` | — | path to MCP servers JSON |
| Provider keys | — | pi's env conventions (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, …) |

## Build and run

```
npm install
npm run build
SHODH_API_KEY=... node dist/index.js
```

Requires Node ≥ 22.19 (pi's floor).

## Deliberately not built

- **Conversation persistence** — transcripts are in-memory per process. pi's
  harness layer ships a session store (JSONL trees, branching, compaction);
  adopting it is a product decision for the seat UI, not harness plumbing.
- **Python/IPython kernel, installable skill packages, recursive sub-agents** —
  external design choices that do not fit this product.
- **A second store for harness behaviors** — harness learnings are memories in
  an isolated scope, on purpose; see above.
