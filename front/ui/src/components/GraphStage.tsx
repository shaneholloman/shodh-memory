import { cn } from "@/lib/utils";
import type { Reachability } from "@/lib/api";
import { RecallDiagram } from "@/components/RecallDiagram";

/**
 * The graph stage.
 *
 * The d3 canvas is ported in a later pass and mounts here. What this file
 * fixes now is the state the stage is in when there is nothing to draw: the
 * previous UI let a raw `graph load failed: Failed to fetch` sit in the middle
 * of the screen, which is the first thing anyone sees when the backend is not
 * up. A designed state that says what is wrong and what to do about it is not
 * decoration — it is the difference between "broken" and "not started yet".
 */

const NODE_LEGEND = [
  { label: "Technology", token: "var(--node-technology)" },
  { label: "Org", token: "var(--node-org)" },
  { label: "Location", token: "var(--node-location)" },
  { label: "Person", token: "var(--node-person)" },
] as const;

/** A server-state notice. Sits above the diagram; never replaces it. */
function StageNotice({ title, body }: { title: string; body: string }) {
  return (
    <div className="border-border bg-card max-w-md rounded-lg border px-4 py-3 text-center">
      <p className="text-[13px] font-medium tracking-tight">{title}</p>
      <p className="text-muted-foreground mt-1 text-[12px] leading-relaxed">
        {body}
      </p>
    </div>
  );
}

export function GraphStage({ reach }: { reach: Reachability }) {
  // No canvas is mounted yet, so there is never a graph to decorate. This flips
  // to real state when the d3 port lands; it is a single flag so the legend and
  // hints cannot get out of step with what is actually drawn.
  const hasGraph = false;

  return (
    <section className="relative min-h-0 flex-1">
      {/* "How this connects" names the graph of your own memory. With no canvas
          mounted the only thing under it is the explainer diagram, which is a
          different claim entirely — it shows how recall works in general, not
          what connects in your data. A label describing absent content is worse
          than no label, so it renders with the canvas. */}
      {hasGraph ? (
        <div className="text-muted-foreground absolute left-4 top-3 z-10 text-[12px]">
          How this connects
        </div>
      ) : null}

      {/* Whatever the connection state, an empty stage is the moment someone is
          working out what this thing does — so the diagram shows in all three.
          A server problem is reported as a notice above it rather than
          replacing the only thing on screen that explains anything. */}
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-6 px-4">
        {reach.state !== "online" ? (
          <StageNotice
            title={
              reach.state === "offline"
                ? "Memory server not running"
                : "The server rejected this key"
            }
            body={
              reach.state === "offline"
                ? "Start the shodh backend and this fills in. Nothing is lost while it is down."
                : "Set SHODH_API_KEY for shodh-front to the key the server was started with."
            }
          />
        ) : null}
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
                <span
                  className="size-2 rounded-full"
                  style={{ background: n.token }}
                />
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
