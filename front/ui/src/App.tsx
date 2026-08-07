import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { probeBackend, type Reachability } from "@/lib/api";
import {
  Sidebar,
  DESTINATIONS,
  type DestinationId,
} from "@/components/Sidebar";
import { GraphStage } from "@/components/GraphStage";
import { Inspector } from "@/components/Inspector";

/**
 * Every control rendered here is wired to something. Three that were not have
 * been removed rather than left inert:
 *
 *  - "LLM-free / Deterministic / On-device" pills. Positioning copy in product
 *    chrome, doing nothing. Those claims belong where someone is deciding
 *    whether to adopt this, not where they are using it.
 *  - An identity dropdown with no menu behind it. The identity is still shown,
 *    as text, until switching users is implemented.
 *  - Graph affordance hints ("scroll to zoom") describing interactions that do
 *    not exist until the canvas is ported. They render with the canvas.
 */

/**
 * Page header.
 *
 * Removing the old tab strip left the three columns with no shared top edge —
 * the search field floated at y=0 against a sidebar whose first item sat 60px
 * lower, and nothing lined up. One header row spanning the content area gives
 * every view the same anchor and is where a view-scoped control (here, search)
 * belongs.
 */
function ViewHeader({
  title,
  children,
}: {
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <header className="border-border flex h-12 shrink-0 items-center gap-3 border-b px-4">
      <h1 className="text-[13px] font-medium tracking-tight">{title}</h1>
      {children ? <div className="ml-auto flex items-center">{children}</div> : null}
    </header>
  );
}

function SearchField({ disabled }: { disabled: boolean }) {
  return (
    <input
      type="search"
      disabled={disabled}
      placeholder="Search memory…"
      aria-label="Search memory"
      className={cn(
        "bg-background border-input placeholder:text-muted-foreground w-[280px]",
        "rounded-md border px-2.5 py-1.5 text-[13px]",
        "focus-visible:border-ring focus-visible:ring-ring/40 focus-visible:outline-none focus-visible:ring-2",
        "disabled:cursor-not-allowed disabled:opacity-50",
      )}
    />
  );
}

/** A destination that has no content yet says so plainly, in its own terms. */
function Placeholder({ id, reach }: { id: DestinationId; reach: Reachability }) {
  const dest = DESTINATIONS.find((d) => d.id === id);
  return (
    <div className="grid flex-1 place-items-center px-8">
      <div className="max-w-sm text-center">
        <p className="text-[15px] font-medium tracking-tight">{dest?.label}</p>
        <p className="text-muted-foreground mt-2 text-[13px] leading-relaxed">
          {dest?.caption}.{" "}
          {reach.state === "online"
            ? "Nothing to show yet."
            : "Start the memory server to see it."}
        </p>
      </div>
    </div>
  );
}

export function App() {
  const [active, setActive] = useState<DestinationId>("recall");
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
    // Recover on its own once the server comes up, instead of needing a reload.
    const timer = window.setInterval(() => void run(), 10_000);

    return () => {
      cancelled = true;
      controller.abort();
      window.clearInterval(timer);
    };
  }, []);

  const online = reach.state === "online";

  return (
    <div className="flex h-screen">
      <Sidebar
        active={active}
        onNavigate={setActive}
        reach={reach}
        identity="defence-live"
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <ViewHeader title={DESTINATIONS.find((d) => d.id === active)!.label}>
          {active === "recall" ? <SearchField disabled={!online} /> : null}
        </ViewHeader>

        <div className="flex min-h-0 flex-1">
          {active === "recall" ? (
            <>
              <div className="border-border flex w-[320px] shrink-0 flex-col border-r">
                <div className="flex min-h-0 flex-1 items-center justify-center p-6">
                  <p className="text-muted-foreground text-center text-[13px] leading-relaxed">
                    {online
                      ? "Search to surface memories, and the chains of cause that surfaced with them."
                      : "Results appear here once the memory server is running."}
                  </p>
                </div>
              </div>
              <GraphStage reach={reach} />
            </>
          ) : (
            <Placeholder id={active} reach={reach} />
          )}
        </div>
      </div>

      <Inspector reach={reach} />
    </div>
  );
}
