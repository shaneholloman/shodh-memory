import { cn } from "@/lib/utils";
import { RecallDiagram } from "./RecallDiagram";

/**
 * The graph stage.
 *
 * The d3 canvas is ported in a later pass and mounts here. What this file
 * fixes now is the state the stage is in when there is nothing to draw: the
 * previous UI let a raw `graph load failed: Failed to fetch` sit in the middle
 * of the screen, which is the first thing anyone sees when the backend is not
 * up. A designed state that says what the product does is not decoration — it
 * is the difference between "broken" and "not started yet".
 *
 * Connection state is not repeated here. It is stated once, in the status strip
 * in the top bar, which is visible from every destination and carries the
 * remedy with it.
 */

const NODE_LEGEND = [
  { label: "Technology", token: "var(--node-technology)" },
  { label: "Org", token: "var(--node-org)" },
  { label: "Location", token: "var(--node-location)" },
  { label: "Person", token: "var(--node-person)" },
] as const;

export function GraphStage() {
  // No canvas is mounted yet, so there is never a graph to decorate. This flips
  // to real state when the d3 port lands; it is a single flag so the legend and
  // hints cannot get out of step with what is actually drawn.
  const hasGraph = false;

  return (
    // `min-w-0`: a flex item's default min-width is its content's intrinsic
    // size, which can silently refuse to shrink below that even when the
    // sibling result column has already claimed the rest of `main`'s box at
    // a narrow viewport. `overflow-hidden` alone doesn't fix that — it only
    // hides what min-width still won't let the box shrink past.
    <section className="relative min-h-0 min-w-0 flex-1 overflow-hidden">
      {/* The floor. Same ruling the canvas will be plotted against, so the
          empty state and the populated one are the same surface. */}
      <div aria-hidden="true" className="graticule pointer-events-none absolute inset-0" />

      {/* "How this connects" names the graph of your own memory. With no canvas
          mounted the only thing under it is the explainer, which is a different
          claim entirely — how recall works in general, not what connects in
          your data. A label describing absent content is worse than no label,
          so it renders with the canvas. */}
      {hasGraph ? (
        <div className="text-muted-foreground absolute top-3 left-4 z-10 text-[12px]">
          How this connects
        </div>
      ) : null}

      {/* `hidden md:flex`, not always-mounted: below ~768px viewport width
          the Inspector and result column's reserved widths (see
          RecallView.tsx / Inspector.tsx) leave this stage under ~135px wide.
          The diagram's own "Docked" explanation text has no minimum-width
          floor, so at that width it wraps character-by-character and its
          vertical overflow escapes past the 92px box it's docked to (the
          ancestor `overflow-hidden` bounds the whole section, not that box).
          RecallDiagram.tsx is explicitly not to be restructured, so the fix
          is here: don't hand it a width it cannot render into. */}
      <div className="absolute inset-0 hidden flex-col items-center justify-center px-8 md:flex">
        <RecallDiagram />
      </div>

      {/* The legend decodes the canvas, so it renders only when there is a
          canvas to decode — a colour key over an empty stage explains nothing.
          The interaction hints ("scroll to zoom") go the same way: they
          describe affordances that do not exist until the graph is ported.

          When it does render, legend and hints share one row with a gap that
          cannot collide. The previous layout overlapped them below ~1500px,
          truncating the hint to "ICK A CLUSTER TO DRILL IN" on a 1440 laptop. */}
      {hasGraph ? (
        <div
          className={cn(
            "absolute inset-x-4 bottom-3 z-10 flex flex-wrap items-center",
            "justify-between gap-x-6 gap-y-2",
          )}
        >
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
            {NODE_LEGEND.map((n) => (
              <span
                key={n.label}
                className="text-muted-foreground flex items-center gap-1.5 text-[11px]"
              >
                <span className="size-2 rounded-full" style={{ background: n.token }} />
                {n.label}
              </span>
            ))}
          </div>
          <span className="text-muted-foreground/70 text-[11px]">
            Scroll to zoom · Drag to pan · Click a cluster to drill in
          </span>
        </div>
      ) : null}
    </section>
  );
}
