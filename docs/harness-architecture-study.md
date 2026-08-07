# prime-agent: what to take, what to refuse

**2026-08-07.** Read of [PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent)
to inform shodh's own conversation seat. We are not adopting it. We are taking
techniques and building them ourselves on
[`earendil-works/pi`](https://github.com/earendil-works/pi) — MIT, TypeScript,
which prime-agent is itself built on ("Our agent and TUI is built on top of
`pi`", prime-agent README).

Everything below is quoted from their own documentation, not inferred.

---

## The central finding

Their Continual Harness stores "supplemental prompts, memories, skill
descriptions, and reusable subagent specifications as durable state", and their
docs draw this line explicitly:

> it's **not a knowledge base retrieval system**, but rather curated,
> reviewable state that the agent actively refines.

That sentence is the whole analysis. **The Continual Harness is what you build
when you do not have associative memory.** Refinement is manual and
evidence-backed, driven by a `/refine` command a person invokes; accumulation is
curation.

shodh's equivalent is not a command. It is Hebbian co-activation, reinforcement
from use, and decay — continuous, automatic, and driven by retrieval rather than
by someone remembering to run `/refine`.

So the instinct to copy the Continual Harness is backwards. It is the workaround
for the absence of the thing we already have. **What we should copy is not the
mechanism but the two properties their mechanism has and ours does not.**

---

## Take: reviewability and rollback

> Records refinement snapshots enabling rollback to prior versions.

> Maintains immutability of base system prompt — changes only affect the
> supplemental layer.

This is the best idea in prime-agent for us, and it is a real gap on our side.

shodh's memory is a single continuously-mutating pool. Hebbian weights move,
importance decays, feedback reinforces — and there is no snapshot, no diff, and
no revert. We cannot currently answer "what did it believe last Tuesday, what
changed since, and why", and we cannot undo a bad update.

For a defence buyer that is not a nice-to-have. Being able to say *this belief
changed on this date, because of these episodes, and here is the state before it*
is the provenance story carried through to its conclusion. It is also a far
better demo than a graph.

Their split is worth stealing wholesale in shape: an **immutable base** plus a
**mutable supplemental layer that is snapshotted**. Ours would be over learned
weights rather than over prompts.

## Take: budgets and gates as first-class

> `/autonomous` continues within configured turn, token, and time budgets and
> can run user-defined quality gates.

This is the right shape for the token-usage surface — a budget that *governs*,
not a counter that reports after the fact. Turn / token / time as three
independent ceilings.

And this line from their docs should become one of our principles verbatim:

> A passed gate checks only what that gate verifies.

That is unusually honest for a README, and it is exactly the discipline our
trace UI needs: a green check must never imply more than it checked. It fits the
positioning — determinism and provenance mean nothing if the UI overclaims.

## Take: persistent goals, because we already have the substrate

> `/goal` keeps objectives active across turns until completion, pause, or
> clear.

We have this already and do not surface it. The retrieval pipeline has a
prospective layer — `prospective_boost` is Layer 4.7 in `ScoreAttribution`,
computed on every recall — plus reminders and todos in the handlers. The
substrate exists; the seat is missing.

Cheap to expose, and it makes recall visibly *anticipatory* rather than
reactive, which no competitor demo does.

## Take: daemon-backed continuity, selectively

> Active sessions, IPython state, schedules, and subagents persist when the
> terminal detaches.

The principle — a conversation is a durable object, not a browser tab — is
right, and it matters for an analyst who closes a laptop mid-investigation. We
want the durability without the kernel. Ours is a persisted conversation with
its retrieved-evidence set, not a live interpreter.

---

## Refuse: everything-is-programmatic

> Everything is programmatic: persistent IPython is the built-in model tool;
> file operations, shell commands, tool use, subagents, and context management
> happen through code.

This is the one to reject hardest, and it is a direct conflict with our product
rather than a matter of taste.

If memory operations execute as Python inside a kernel, they render as opaque
`execute` cells. **The traceability surface *is* the product** — "looked up 4
memories, here is why each scored where it did" is the differentiator. Burying
that inside a REPL transcript destroys the only screen that shows what we have
and others do not.

Consequences we also avoid by refusing it: an IPython kernel running
model-generated code at user permissions (their docs are explicit that it is
"not a security sandbox"), and one-session-per-connection, which forces a
process per conversation.

## Refuse: skills as importable Python packages

> Skills are importable Python packages, and the built-in skill creator can turn
> recurring workflows into project or personal skills.

Sound idea, wrong substrate. It ties skill accumulation to the Python kernel we
just refused. shodh is Rust plus TypeScript. If we want accumulating
capabilities later, they should accumulate as *memory* — which is the asset we
actually have — not as installable packages.

## Refuse: the RLM, for now

> The Recursive Language Model (RLM) treats context as variables
> (*prompt-as-a-variable*) and tools like recursive subagents as function calls
> inside a persistent REPL.

This is what produced their ARC-AGI-3 result. It is a *reasoning* technique,
and our demo is a *retrieval* demo — needle-in-haystack over a corpus larger
than context. Recursive subagents do not find needles; retrieval does.

Adopting it would mean paying the kernel, the session limits and the broker for
a capability the demo never exercises. Their own headline number is also
qualified in our notes: community board, self-reported, and ARC Prize's official
board excludes harness-driven scores.

Revisit if the product ever needs multi-step autonomous reasoning. Not for this.

---

## What we build on `pi`

`pi` is the substrate for the parts that are undifferentiated work:

| Surface | Package | Why it is not ours to build |
|---|---|---|
| Model selection, incl. local | `@earendil-works/pi-ai` — "unified multi-provider LLM API (OpenAI, Anthropic, Google, …)" | This *is* BYOM. Building a provider abstraction ourselves buys nothing. |
| Agent loop, tool calling, state | `@earendil-works/pi-agent-core` — "agent runtime with tool calling and state management" | Drivable programmatically rather than as a TUI, which is what embedding requires. |
| Token accounting | usage reported per call by the provider layer | Feeds the budget ceilings above. |

Credentials are provider credentials held server-side, never in the browser.
Note this is **not** "sign in" in the account sense — shodh stays local-first,
and the profile control switches *whose memory* is on screen, which is a
different promise.

### Unverified, and build-blocking

Neither README confirmed these, and both decide the design:

1. **Streaming in `pi-agent-core`.** A conversation UI without token streaming
   is a spinner, and local models make that worse. If it is absent, that changes
   the plan rather than the schedule.
2. **MCP client maturity in `pi-agent-core`.** shodh-memory reaches the seat over
   MCP; the README only gestures at MCP support.
3. **Local providers in `pi-ai`.** Ollama exposes an OpenAI-compatible endpoint,
   so a base-URL override probably suffices — probably is not verified.

Read the source on these three before writing seat code.

---

## Net

Take from prime-agent: **snapshot-and-rollback over a mutable layer**, **budgets
and gates as first-class**, **persistent goals**, **conversations as durable
objects**.

Refuse: **the kernel**, **Python skills**, **the RLM**.

The asymmetry is the point. Their durable state is curated by hand because they
have no retrieval system. Ours is learned from use because retrieval is what we
are. We should take their *discipline* about that state — versioned, reviewable,
revertible — and not their *mechanism* for filling it.
