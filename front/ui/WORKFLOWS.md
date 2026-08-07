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

The shell renders these destinations. **None of the chains are wired** — there
is no selection state, no Inspector content, and no graph. Building them
requires reading the handlers for recall, sessions, anomalies and todos first;
no response shape in this app is guessed.
