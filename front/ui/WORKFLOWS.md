# The workflows this UI has to support

The dashboard's problem is not styling. It is that every destination is a dead
end: you arrive, you read a number, and there is nowhere to go. A person cannot
finish a job in it.

What follows is the spine. It is written down because the layout only makes
sense once you know which chains it has to carry.

## The one rule

**There is a single object-detail surface — the Inspector — and every list in
the product feeds it.** A memory, an entity, a session, an anomaly and a task
all open in the same pane, and that pane always offers the next hop. This is
what makes a click lead somewhere instead of terminating.

## Chain 1 — "What do we know about X?"

    search → result → Inspector(memory)
                        ├→ entity mentioned      → graph focuses it
                        ├→ session it came from  → Chain 2
                        └→ what it activated     → back into results

Selecting a result highlights its nodes in the graph; clicking a node filters
the results to memories touching it. The list and the graph are two views of
one selection, not two panels that happen to share a screen.

## Chain 2 — "Why does the system believe this?"

    Inspector(memory) → source episodes → Inspector(session)
                                            ├→ when, which agent wrote it
                                            └→ everything else from that session

This is the trust chain. It is the differentiator: every belief traces to the
sessions it came from, and a person must be able to walk that without help.

## Chain 3 — "What should I look at?"

    Anomalies → deviation row (z-score + its deterministic explanation)
              → Inspector(memory that caused it)
              → Chain 1 or Chain 2
              → mark reviewed  (POST /api/reinforce — verify handler first)

Anomalies is the only destination that proposes work. It has to hand off into
the same detail surface, or it is a list of numbers nobody acts on.

## Chain 4 — "What is outstanding?"

    Tasks → task → the session that captured it → Chain 2

## What this means for the layout

- The Inspector is the most important pane, not the third column of stats it
  used to be. It is never empty when something is selected, and it always shows
  at least one onward link.
- Selection is global state, not per-panel. One selected object at a time.
- Back/forward has to work: walking a chain and returning is the core motion.

## Status

Last checked against the running app, not inferred.

**The conversation seat is the primary surface.** `/chat` is home, and the seat
is where a person actually works: the model answers with memory on the table
and the operations it ran are shown inline. Recall is its deterministic peer —
the same corpus reached without a model in the loop, which is what makes an
answer checkable rather than merely fluent. Both feed one Inspector.

**Chain 1 — wired.** Search returns results; selecting one opens it in the
Inspector with its content, entities, tier and score attribution. The graph
canvas plots the same result set and clicking a node selects the same memory,
so the list and the canvas are two views of one selection.

**Chain 2 — wired.** The Inspector's "What it connects to" walks
`RecallResponse.lineage`, and selecting an edge moves the Inspector to the
other end of it. Each edge carries the server's own relation type and
confidence. The hop is memory → memory: the server returns causal edges
between recalled memories directly, which is the trust chain the product
promises. Walking back to *sessions* is not built — see below.

**Geo — wired, on the same selection.** `/geo` plots the geotagged part of the
current recall result on a vendored world basemap; clicking a point selects
that memory in the same Inspector. Coordinates are caller-supplied and nothing
in the Rust path derives them, so most results have none and the view says so.

**Chain 3 — not built.** `/anomalies` is a placeholder. It needs a real
deviation feed before it can propose work.

**Chain 4 — partially built, and blocked on the data model rather than on
effort.** Tasks is a real list against `POST /api/todos`, but it cannot hand
off to the Inspector: `Todo` (src/memory/types.rs:3646) carries no session
reference — only `related_memory_ids` — and the Inspector reads the recall
cache, which a todo was never part of. Wiring `select()` there would open the
Inspector on an absent cache entry. It stays a standalone list until a todo
can name where it came from.

**Sessions are the missing object.** Both Chain 2's second hop and Chain 4
terminate at "the session that captured this", and there is no session detail
surface yet. That is the next thing worth building, not more list views.

No response shape in this app is guessed; each wire type cites the Rust struct
it mirrors, and anything not in the handler is not in the UI.
