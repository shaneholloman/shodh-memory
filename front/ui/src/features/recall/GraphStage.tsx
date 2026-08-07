import { cn } from "@/lib/utils";
import type { Reachability } from "@/lib/api";
import { RecallDiagram } from "./RecallDiagram";
import { GraphCanvas, useMemoryTypes } from "./GraphCanvas";
import { useRecall } from "./useRecall";

/**
 * The graph stage.
 *
 * Two states, and which one shows is decided by whether there is anything to
 * draw rather than by whether the backend is up. With no results the stage
 * carries RecallDiagram — an explanation of how recall works, which is a claim
 * about the product in general. Once a recall returns memories the canvas
 * mounts and the label above it ("How this connects") becomes true of the data
 * on screen, which is a different and much stronger claim. A label describing
 * absent content is worse than no label, so both it and the legend render with
 * the canvas and not before.
 *
 * Connection state is not repeated here. It is stated once, in the status strip
 * in the top bar, which is visible from every destination and carries the
 * remedy with it.
 */

export function GraphStage({ reach }: { reach: Reachability }) {
  const { data } = useRecall(reach);
  const memories = data?.memories ?? [];
  const lineage = data?.lineage ?? [];
  const types = useMemoryTypes(memories);
  const hasGraph = memories.length > 0;

  const edgeCount = lineage.length;

  return (
    // `min-w-0`: a flex item's default min-width is its content's intrinsic
    // size, which can silently refuse to shrink below that even when the
    // sibling result column has already claimed the rest of `main`'s box at
    // a narrow viewport. `overflow-hidden` alone doesn't fix that — it only
    // hides what min-width still won't let the box shrink past.
    <section className="relative min-h-0 min-w-0 flex-1 overflow-hidden">
      {/* The floor. Same ruling the canvas is plotted against, so the
          empty state and the populated one are the same surface. */}
      <div aria-hidden="true" className="graticule pointer-events-none absolute inset-0" />

      {hasGraph ? (
        <>
          <GraphCanvas memories={memories} lineage={lineage} />
          <div className="text-muted-foreground pointer-events-none absolute top-3 left-4 z-10 text-[12px]">
            How this connects
          </div>
        </>
      ) : (
        /* `hidden md:flex`, not always-mounted: below ~768px viewport width
           the Inspector and result column's reserved widths (see
           RecallView.tsx / Inspector.tsx) leave this stage under ~135px wide.
           The diagram's own "Docked" explanation text has no minimum-width
           floor, so at that width it wraps character-by-character and its
           vertical overflow escapes past the 92px box it's docked to (the
           ancestor `overflow-hidden` bounds the whole section, not that box).
           RecallDiagram.tsx is explicitly not to be restructured, so the fix
           is here: don't hand it a width it cannot render into. */
        <div className="absolute inset-0 hidden flex-col items-center justify-center px-8 md:flex">
          <RecallDiagram />
        </div>
      )}

      {/* The legend decodes the canvas, so it renders only when there is a
          canvas to decode — a colour key over an empty stage explains nothing.

          Legend and hints share one row with a gap that cannot collide. An
          earlier layout overlapped them below ~1500px, truncating the hint to
          "ICK A CLUSTER TO DRILL IN" on a 1440 laptop. */}
      {hasGraph ? (
        <div
          className={cn(
            "pointer-events-none absolute inset-x-4 bottom-3 z-10 flex flex-wrap items-center",
            "justify-between gap-x-6 gap-y-2",
          )}
        >
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
            {/* The legend is the memory types actually present, in the order
                the canvas assigns hues in. `memory_type` is a free-form string
                server-side (src/handlers/types.rs:258) with no enum behind it,
                so a fixed legend would be an invented ontology — it would name
                categories this corpus may not contain and omit ones it does. */}
            {types.map((t, i) => (
              <span
                key={t}
                className="text-muted-foreground flex items-center gap-1.5 text-[11px]"
              >
                <span
                  className="size-2 rounded-full"
                  style={{ background: `var(--chart-${(i % 5) + 1})` }}
                />
                {t}
              </span>
            ))}
          </div>
          <span className="text-muted-foreground/70 text-[11px]">
            {/* Say what the edges ARE, not just how to move the camera. Zero
                causal edges across a result set is a finding about the corpus,
                not a broken canvas, and staying silent about it invites the
                opposite reading. */}
            {edgeCount > 0
              ? `${edgeCount} causal ${edgeCount === 1 ? "edge" : "edges"} · scroll to zoom · drag to pan · click a node to inspect`
              : "No causal edges connect these results · scroll to zoom · drag to pan · click a node to inspect"}
          </span>
        </div>
      ) : null}
    </section>
  );
}
