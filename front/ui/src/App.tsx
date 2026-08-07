import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { probeBackend, type Reachability } from "@/lib/api";
import { TopBar } from "@/components/TopBar";
import { GraphStage } from "@/components/GraphStage";
import { SidePanel, PanelEmpty } from "@/components/SidePanel";

const VIEWS = ["Live", "Anomalies", "Work"] as const;
type View = (typeof VIEWS)[number];

/** Segmented nav. Active state is a filled surface, not an underline + accent. */
function ViewTabs({
  view,
  onChange,
}: {
  view: View;
  onChange: (v: View) => void;
}) {
  return (
    <nav className="flex items-center gap-1">
      {VIEWS.map((v) => (
        <button
          key={v}
          type="button"
          onClick={() => onChange(v)}
          aria-current={v === view ? "page" : undefined}
          className={cn(
            "rounded-md px-2.5 py-1 text-[12px] font-medium transition-colors duration-100",
            "focus-visible:ring-ring focus-visible:outline-none focus-visible:ring-2",
            v === view
              ? "bg-accent text-accent-foreground"
              : "text-muted-foreground hover:text-foreground hover:bg-accent/60",
          )}
        >
          {v}
        </button>
      ))}
    </nav>
  );
}

function CueBar() {
  return (
    <div className="border-border border-b p-3">
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="Cue a recall…"
          aria-label="Cue a recall"
          className={cn(
            "bg-card border-input placeholder:text-muted-foreground flex-1",
            "rounded-md border px-2.5 py-2 text-[13px]",
            "focus-visible:border-ring focus-visible:ring-ring/40 focus-visible:outline-none focus-visible:ring-2",
          )}
        />
        <button
          type="button"
          className={cn(
            "bg-primary text-primary-foreground rounded-md px-3 text-[13px] font-medium",
            "hover:bg-primary/90 transition-colors duration-100",
            "focus-visible:ring-ring focus-visible:outline-none focus-visible:ring-2",
          )}
        >
          Recall
        </button>
      </div>
      <p className="text-muted-foreground mt-2 text-[11px] leading-relaxed">
        One cue surfaces a cascade of associated memories, plus the causal
        chains that lit up with them.
      </p>
    </div>
  );
}

export function App() {
  const [view, setView] = useState<View>("Live");
  const [reach, setReach] = useState<Reachability>({
    state: "offline",
    detail: "not probed yet",
  });

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    const run = async () => {
      const result = await probeBackend(controller.signal);
      if (!cancelled) setReach(result);
    };

    void run();
    // Re-probe so the dashboard recovers on its own once the server comes up,
    // rather than needing a reload.
    const timer = window.setInterval(() => void run(), 10_000);

    return () => {
      cancelled = true;
      controller.abort();
      window.clearInterval(timer);
    };
  }, []);

  return (
    <div className="flex h-screen flex-col">
      <TopBar reach={reach} identity="defence-live" />

      <div className="border-border flex h-10 flex-shrink-0 items-center border-b px-3">
        <ViewTabs view={view} onChange={setView} />
      </div>

      <main className="grid min-h-0 flex-1 grid-cols-[340px_1fr_300px]">
        <div className="border-border bg-card flex min-h-0 flex-col border-r">
          <CueBar />
          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            <p className="text-[13px] font-medium tracking-tight">
              The river is still
            </p>
            <p className="text-muted-foreground mt-2 text-[13px] leading-relaxed">
              Each recall blooms into the memories it surfaced and the causal
              chains that lit up with them — spreading activation, made visible.
            </p>
          </div>
        </div>

        <GraphStage reach={reach} />

        <aside className="border-border bg-card min-h-0 overflow-y-auto border-l">
          <SidePanel
            title="Memory tiers"
            hint="How much is held: working, session, long-term."
          >
            <PanelEmpty>Counts appear once the server is reachable.</PanelEmpty>
          </SidePanel>
          <SidePanel
            title="Graph"
            hint="Entities and typed relations learned from text — no LLM."
          >
            <PanelEmpty>No graph loaded.</PanelEmpty>
          </SidePanel>
          <SidePanel
            title="Connected clients"
            hint="Sessions writing to this memory right now."
          >
            <PanelEmpty>Nothing connected.</PanelEmpty>
          </SidePanel>
          <SidePanel
            title="Detail"
            hint="Every belief traces to its source episodes."
          >
            <PanelEmpty>
              Select a node, an edge, or a surfaced memory.
            </PanelEmpty>
          </SidePanel>
        </aside>
      </main>
    </div>
  );
}
