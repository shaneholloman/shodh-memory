# Visual direction

**Decided 2026-08-07: adopt the Gridline Dashboard structure. This supersedes
the Linear direction.**

Both were real directions and they are not compatible. Linear is sans-serif,
labelled text navigation, calm indigo, list-first. Gridline is monospace body
copy, an icon rail, teal accent, and a full-bleed spatial canvas. Taking half of
each gets neither. The Linear token layer in `src/index.css` stays as the
mechanism — semantic shadcn variables — but its *values* are now up for change.

## Why Gridline

Judged from the live preview, not from a description:

- **The map is the interaction shodh's graph needs.** Region pins that lazy-load
  on click is the same motion as "click a cluster to drill in", and it is what
  the geotemporal work wants.
- **`Published / Validated / In Review` plus `v2.4.1 → v2.4.2`** in the header is
  a review-state and version-diff affordance. That maps almost directly onto
  belief drift and provenance, which is the differentiator.
- **Progressive disclosure is already its idiom** — every pin says "Click to
  load" rather than showing everything at once.

## Changes to make, not copy verbatim

1. **Smaller icons** in the rail. Gridline's are oversized for the density this
   product needs.
2. **Hover expands the rail** into a labelled column. Icon-only navigation fails
   the unaided-decipherability test — seven unlabelled glyphs is a memory game.
   Requirements:
   - Expand must **overlay**, not push content. Reflowing a map or a graph on
     mouse-over is disorienting and re-lays-out the thing being pointed at.
   - Keyboard must reach it: focus expands the same as hover, and labels are in
     the DOM at all times so screen readers are never given bare icons. Use
     `aria-label` on every control regardless.
   - Honour `prefers-reduced-motion` — the expand is a width transition and must
     collapse to an instant state change.
   - A short close delay, so crossing the rail diagonally toward content does not
     flicker it shut.

## Source

Captured from `ui.watermelon.sh/dashboard/gridline-dashboard` (Source Code tab)
into `scratchpad/gridline-src.json` — 13 files, ~72 kB. The registry endpoint
`registry.watermelon.sh/r/gridline-dashboard.json` returns the SPA 404 page as
HTML, so `shadcn add` cannot fetch it; the site advertises a command that does
not work. The files are lifted from that JSON instead.

## Unverified before this lands

- Its npm dependencies (Radix set, `@tabler/icons-react`) and which map library
  it uses — not yet identified.
- Whether its components read shadcn's semantic variables or hardcode their own
  colours. If they hardcode, the token layer buys nothing and each component
  needs restyling. **This is still the untested premise the whole approach rests
  on** and should be checked with one component before porting the rest.
