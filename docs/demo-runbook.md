# Workbench demonstration runbook

A repeatable, verifiable walk of every surface, in the order the beats build on
each other. Each beat names what it proves and what "working" looks like, so a
rehearsal produces a pass/fail list rather than an impression.

## Preflight

1. **Binary** built from the merged demo branch; `models/` present beside it
   (GLiNER + spaCy assets are embedded/provisioned as of the causal-extraction
   fix — a scratch deployment must produce typed causal edges; if it does not,
   stop here, nothing downstream is worth rehearsing).
2. **Fresh data dir.** Never demo on a store that carries earlier experiments:
   `SHODH_MEMORY_PATH=<fresh> SHODH_API_KEYS=<key> SHODH_PORT=3030 shodh-memory-server`
3. **Seed:** `SHODH_API_URL=... SHODH_API_KEY=... USER_ID=demo node seat/eval/seed-demo-corpus.mjs`
   — the smoke line must report lineage edges > 0.
4. **Seat** on its default data dir (holds the provider credential from
   sign-in): `SHODH_API_URL=... SHODH_API_KEY=... node seat/dist/index.js`
5. **Front**: the embedded UI from the binary, or `vite` in `front/ui` for a
   dev walk. Providers page must show the model provider READY before starting.
6. Machine hygiene: antivirus exclusion for the build/binary directory
   (a debug exe binding ports repeatedly has been deleted from disk before);
   close file-sync apps; external display mirrored at 1440×900.

## The beats

| # | Beat | Proves | Working looks like |
|---|---|---|---|
| 1 | Recall: "why did the bridge collapse" | deterministic retrieval, no model | ranked results instantly; unrelated island absent |
| 2 | Select result → Inspector | attribution | 20-factor breakdown; moved factors listed, rest counted |
| 3 | Lineage hop in Inspector | causal chains | typed edges with confidence; hop lands on the cause |
| 4 | Graph destination | the ontology, visibly | typed node colours (person/org/location/…); three edge-tier colours with counts; declared edge floor caption |
| 5 | Entity click → provenance | trust chain | entity → relations → source memory text |
| 6 | Geo: search "Baltimore" | located memory | pins at real coordinates; click → Inspector |
| 7 | Floating seat over Graph | conversation follows the analyst | minimized bar shows logo · model · live tokens · billing badge |
| 8 | Ask: "what connects the Dali to the port closure? cite memories" | grounded generation | evidence fills BEFORE the model speaks; `[mem:id]` chips in the answer; reinforcement line with revert |
| 9 | Mid-conversation model swap | BYOM / evidence independence | transcript and evidence untouched; new model badge; billing semantics switch honestly |
| 10 | Tasks | work capture | seeded todos, priority/overdue badges |
| 11 | Learning ledger (via chat blocks) | reviewability | reinforce/write events listed; revert works and says what it can and cannot undo |

Beat 9 needs a second provider configured (local runtime installed and a model
pulled, or a second hosted credential). If neither exists on the demo machine,
cut the beat deliberately in rehearsal — do not improvise it live.

## Rehearsal rule

Walk all beats twice: once as the operator, once letting someone else drive
from this table without help. Anything that needs explaining is a finding.
