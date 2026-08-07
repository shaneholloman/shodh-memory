import { cn } from "@/lib/utils";
import type { Reachability } from "@/lib/api";

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

function StageMessage({
  title,
  body,
  action,
}: {
  title: string;
  body: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="absolute inset-0 grid place-items-center px-8">
      <div className="max-w-sm text-center">
        <p className="text-[15px] font-medium tracking-tight">{title}</p>
        <p className="text-muted-foreground mt-2 text-[13px] leading-relaxed">
          {body}
        </p>
        {action ? <div className="mt-4">{action}</div> : null}
      </div>
    </div>
  );
}

export function GraphStage({ reach }: { reach: Reachability }) {
  return (
    <section className="relative min-h-0 flex-1">
      <div className="text-muted-foreground absolute left-4 top-3 z-10 text-[12px]">
        Knowledge graph
      </div>

      {reach.state === "offline" ? (
        <StageMessage
          title="No memory server"
          body="Start the shodh backend and this fills in. Nothing is lost while it is down — the graph is rebuilt from what the server already holds."
        />
      ) : reach.state === "unauthorized" ? (
        <StageMessage
          title="The server rejected this key"
          body="The dashboard reached the memory server, but its API key was refused. Set SHODH_API_KEY for shodh-front to the key the server was started with."
        />
      ) : (
        <StageMessage
          title="Nothing recalled yet"
          body="Cue a recall and the memories it surfaces — with the causal chains that lit up alongside them — are drawn here."
        />
      )}

      {/* Legend and canvas hints share one row with a gap that cannot collide:
          the previous layout overlapped them below ~1500px, truncating the
          hint text to "ICK A CLUSTER TO DRILL IN" on a 1440 laptop. */}
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
    </section>
  );
}
