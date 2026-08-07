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
 * The shell.
 *
 * Nothing here is in normal flow except the content itself: the rail, the
 * header and the Inspector are all absolutely positioned, and `main` is offset
 * by their widths. That is Gridline's structure and it is not decoration —
 * it is what lets the rail expand over the stage instead of shoving it, and
 * what will let the graph canvas take the full width with panels floated on
 * top once it is ported. A flex row cannot do either.
 *
 * Two things from Gridline are deliberately not here. Its theme switch and the
 * MutationObserver behind it are gone: this product is dark-only (index.css
 * defines one set of values), so the control would toggle nothing. Its version
 * strip is gone for the same reason — there is no version feed behind it, and
 * chrome that displays invented data is worse than chrome that is absent.
 *
 * Also removed earlier, and staying removed:
 *  - "LLM-free / Deterministic / On-device" pills. Positioning copy in product
 *    chrome, doing nothing. Those claims belong where someone is deciding
 *    whether to adopt this, not where they are using it.
 *  - An identity dropdown with no menu behind it. The identity is still shown,
 *    as text, until switching users is implemented.
 *  - Graph affordance hints ("scroll to zoom") describing interactions that do
 *    not exist until the canvas is ported. They render with the canvas.
 */

/** Collapsed rail width and Inspector width, as the offsets `main` reserves.
 *  The rail's *expanded* width is deliberately absent: reserving it would make
 *  the expansion push content, which is the thing it must not do. */
const RAIL_OFFSET = "pl-14";
const INSPECTOR_OFFSET = "pr-[280px]";

/**
 * Connection state, said once for the whole product.
 *
 * Gridline's segmented strip, carrying the one piece of live state this app
 * actually has. It replaces three separate statements of the same fact — a
 * card in the middle of the stage, a row at the foot of the rail, and a line
 * in the Inspector — which between them made an ordinary not-started-yet state
 * read as something broken.
 *
 * A state and its remedy travel together. "Offline" on its own sends people to
 * check their network when the server simply is not running.
 */
function StatusStrip({ reach }: { reach: Reachability }) {
  const { tone, state, remedy } =
    reach.state === "online"
      ? {
          tone: "text-[var(--live)]",
          state: "Connected",
          remedy: "Memory server reachable",
        }
      : reach.state === "unauthorized"
        ? {
            tone: "text-destructive",
            state: "Key rejected",
            remedy: `Server answered ${reach.status} — set SHODH_API_KEY to the key it was started with`,
          }
        : {
            tone: "text-warn",
            state: "Not running",
            remedy: "Start the shodh backend — nothing is lost while it is down",
          };

  return (
    <div className="bg-muted mono hidden h-[26px] min-w-0 shrink items-center overflow-hidden rounded-md text-[11px] sm:flex">
      <span
        className={cn(
          "border-sidebar flex h-full shrink-0 items-center gap-2 border-r-2 px-2.5",
          tone,
        )}
      >
        <span className="size-1.5 shrink-0 rounded-full bg-current" />
        {state}
      </span>
      <span className="text-muted-foreground hidden h-full min-w-0 items-center px-2.5 lg:flex">
        <span className="truncate">{remedy}</span>
      </span>
    </div>
  );
}

function TopBar({
  title,
  reach,
  children,
}: {
  title: string;
  reach: Reachability;
  children?: React.ReactNode;
}) {
  return (
    <header
      className={cn(
        "border-sidebar-border bg-sidebar text-sidebar-foreground",
        "absolute inset-x-0 top-0 z-20 flex h-12 items-center gap-3 border-b px-4",
        RAIL_OFFSET,
      )}
    >
      {/* Everything but the title sits right. The rail expands over this bar
          from the left, so anything in its first 224px gets occluded on hover
          — and a status strip sliced mid-word reads as a rendering fault. The
          title is the one thing that can afford to go: while the rail is open
          it is showing its own header, which names the product anyway. */}
      <h1 className="shrink-0 text-[13px] font-medium tracking-tight">{title}</h1>
      <div className="ml-auto flex min-w-0 items-center gap-3 pl-3">
        <StatusStrip reach={reach} />
        {children}
      </div>
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
        // Shrinks rather than pushing the title off. A fixed 280px crowded the
        // heading out of the bar below ~480px.
        "bg-background border-input placeholder:text-muted-foreground w-full min-w-0 max-w-[280px]",
        "rounded-md border px-2.5 py-1.5 text-[13px]",
        "focus-visible:border-ring focus-visible:ring-ring/40 focus-visible:ring-2 focus-visible:outline-none",
        "disabled:cursor-not-allowed disabled:opacity-50",
      )}
    />
  );
}

/** A destination that has no content yet says so plainly, in its own terms. */
function Placeholder({ id, reach }: { id: DestinationId; reach: Reachability }) {
  const dest = DESTINATIONS.find((d) => d.id === id);
  return (
    <div className="grid h-full place-items-center px-8">
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
    <div className="relative h-svh min-h-0 overflow-hidden">
      <Sidebar active={active} onNavigate={setActive} identity="defence-live" />

      <TopBar title={DESTINATIONS.find((d) => d.id === active)!.label} reach={reach}>
        {active === "recall" ? <SearchField disabled={!online} /> : null}
      </TopBar>

      <main
        className={cn("h-full pt-12", RAIL_OFFSET, online && INSPECTOR_OFFSET)}
      >
        {active === "recall" ? (
          <div className="flex h-full min-h-0">
            {online ? (
              <div className="border-border flex w-[320px] shrink-0 flex-col border-r">
                <div className="flex min-h-0 flex-1 items-center justify-center p-6">
                  <p className="text-muted-foreground text-center text-[13px] leading-relaxed">
                    Search to surface memories, and the chains of cause that
                    surfaced with them.
                  </p>
                </div>
              </div>
            ) : null}
            <GraphStage />
          </div>
        ) : (
          <Placeholder id={active} reach={reach} />
        )}
      </main>

      {/* Only when the product is usable. With the server down there is nothing
          to select, so an Inspector can only repeat what the stage already
          says — and the offline screen had four panels apologising separately
          for one problem. One statement of a problem is information; four is
          noise, and it made an ordinary not-started-yet state read as broken. */}
      {online ? <Inspector reach={reach} /> : null}
    </div>
  );
}
