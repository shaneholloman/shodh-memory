import type { Reachability } from "@/lib/api";

/**
 * The right rail.
 *
 * Was four permanent stat panels — Memory tiers / Graph / Connected LLMs /
 * Detail — which meant that with nothing selected the rail read as four
 * identical "unavailable" lines, and with something selected the thing you
 * cared about was the fourth item down.
 *
 * It is now one contextual surface answering "what am I looking at". With no
 * selection it explains what this pane is for rather than showing empty
 * counters. Real counts land here once their handlers are read; nothing is
 * invented in the meantime.
 */
export function Inspector({ reach }: { reach: Reachability }) {
  return (
    <aside className="border-border bg-card w-[280px] shrink-0 overflow-y-auto border-l">
      <header className="border-border border-b px-4 py-3">
        <h2 className="text-[12px] font-medium tracking-tight">Detail</h2>
        <p className="text-muted-foreground mt-0.5 text-[11px] leading-relaxed">
          Every belief traces back to the sessions it came from.
        </p>
      </header>

      <div className="px-4 py-3">
        {reach.state === "online" ? (
          <p className="text-muted-foreground text-[12px] leading-relaxed">
            Select a memory or a node in the graph to see where it came from —
            which session recorded it, when, and what else it activated.
          </p>
        ) : (
          <p className="text-muted-foreground text-[12px] leading-relaxed">
            Detail becomes available once the memory server is running.
          </p>
        )}
      </div>
    </aside>
  );
}
