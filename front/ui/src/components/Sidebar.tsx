import { useCallback, useEffect, useRef, useState } from "react";
import { Search, TriangleAlert, ListChecks } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Reachability } from "@/lib/api";
import shodhMark from "@/assets/shodh-mark.png";

/**
 * Primary navigation — an icon rail that expands on hover or focus.
 *
 * Names are what the thing is, not what the subsystem is called:
 *   Recall     — the search surface (was "Live")
 *   Anomalies  — deviations from this user's own baseline (was unqualified)
 *   Tasks      — the todo list (was "Work", which named nothing)
 *
 * The rail is Gridline's structure with two deliberate departures. Its icons
 * are sized for a far sparser screen than this one, so they are smaller here.
 * And Gridline leaves the rail icon-only, hanging labels off tooltips — seven
 * unlabelled glyphs is a memory game, and a tooltip only pays out after you
 * have already guessed which one to point at. Expanding the whole column
 * instead shows every label at once, which is the question a person actually
 * has ("which of these is the one I want?").
 *
 * Three things this has to get right, none of them optional:
 *
 *  - It expands as an OVERLAY. The stage behind it holds a graph; reflowing a
 *    force layout because a pointer crossed the edge of the screen is
 *    disorienting, and it re-lays-out the very thing being pointed at.
 *  - Focus expands it exactly as hover does, and the labels are in the DOM at
 *    all times — clipped, never absent — so a screen reader is never handed a
 *    column of bare icons.
 *  - Leaving is delayed. Pointers travel diagonally toward content and clip
 *    the rail's corner on the way out; closing on the first mouseleave makes
 *    it flicker.
 *
 * Reduced motion is handled globally in index.css, which collapses every
 * transition to 0.01ms — this component needs no separate branch for it.
 */

export const DESTINATIONS = [
  {
    id: "recall",
    label: "Recall",
    icon: Search,
    caption: "Search memory and see what connects",
  },
  {
    id: "anomalies",
    label: "Anomalies",
    icon: TriangleAlert,
    caption: "What deviates from this user's baseline",
  },
  {
    id: "tasks",
    label: "Tasks",
    icon: ListChecks,
    caption: "Open work captured from sessions",
  },
] as const;

export type DestinationId = (typeof DESTINATIONS)[number]["id"];

const CLOSE_DELAY_MS = 180;

/**
 * Label text that is clipped rather than removed when the rail is collapsed.
 * Kept as one component so no caller can accidentally unmount a label and
 * leave an icon with no accessible name.
 */
function RailLabel({
  open,
  children,
  className,
}: {
  open: boolean;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "truncate transition-opacity duration-150",
        open ? "opacity-100" : "opacity-0",
        className,
      )}
    >
      {children}
    </span>
  );
}

function ConnectionRow({ reach, open }: { reach: Reachability; open: boolean }) {
  const tone =
    reach.state === "online"
      ? "bg-[var(--live)]"
      : reach.state === "unauthorized"
        ? "bg-destructive"
        : "bg-muted-foreground/40";

  // Say what is wrong, in the words of the thing that is wrong. "Offline" alone
  // sends people to check their network when the server simply is not running.
  const label =
    reach.state === "online"
      ? "Memory server connected"
      : reach.state === "unauthorized"
        ? "Memory server rejected the key"
        : "Memory server not running";

  return (
    <div className="text-muted-foreground flex h-8 items-center gap-3 px-[1.15rem] text-[11px]">
      <span className={cn("size-1.5 shrink-0 rounded-full", tone)} />
      <RailLabel open={open}>{label}</RailLabel>
    </div>
  );
}

export function Sidebar({
  active,
  onNavigate,
  reach,
  identity,
}: {
  active: DestinationId;
  onNavigate: (id: DestinationId) => void;
  reach: Reachability;
  identity: string;
}) {
  const [open, setOpen] = useState(false);
  const closeTimer = useRef<number | undefined>(undefined);

  const cancelClose = useCallback(() => {
    if (closeTimer.current !== undefined) {
      window.clearTimeout(closeTimer.current);
      closeTimer.current = undefined;
    }
  }, []);

  const openRail = useCallback(() => {
    cancelClose();
    setOpen(true);
  }, [cancelClose]);

  const closeRail = useCallback(() => {
    cancelClose();
    closeTimer.current = window.setTimeout(() => setOpen(false), CLOSE_DELAY_MS);
  }, [cancelClose]);

  useEffect(() => cancelClose, [cancelClose]);

  return (
    <aside
      aria-label="Primary navigation"
      // Focus events bubble (unlike focusin's absence on React's synthetic
      // `onFocus` — React's onFocus DOES bubble), so one handler on the
      // container covers every control inside without wiring each one.
      onMouseEnter={openRail}
      onMouseLeave={closeRail}
      onFocus={openRail}
      // A blur that lands on another control inside the rail is not a leave.
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
          closeRail();
        }
      }}
      // Escape closes it for keyboard users who expanded it by tabbing in and
      // want it out of the way without leaving the rail.
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          cancelClose();
          setOpen(false);
        }
      }}
      className={cn(
        "border-sidebar-border bg-sidebar text-sidebar-foreground",
        "absolute inset-y-0 left-0 z-30 flex flex-col overflow-hidden border-r",
        "transition-[width] duration-200 ease-out",
        open ? "w-56 shadow-2xl shadow-black/40" : "w-14",
      )}
    >
      <div className="border-sidebar-border flex h-12 shrink-0 items-center gap-3 border-b px-[0.9rem]">
        <img
          src={shodhMark}
          alt=""
          aria-hidden="true"
          className="size-6 shrink-0 object-contain"
        />
        <div className="min-w-0 leading-tight">
          <RailLabel open={open} className="block text-[13px] font-semibold tracking-tight">
            shodh
          </RailLabel>
          <RailLabel open={open} className="text-muted-foreground block text-[11px]">
            {identity}
          </RailLabel>
        </div>
      </div>

      <nav className="flex flex-col gap-1 p-2">
        {DESTINATIONS.map((d) => {
          const Icon = d.icon;
          const isActive = d.id === active;
          return (
            <button
              key={d.id}
              type="button"
              onClick={() => onNavigate(d.id)}
              aria-current={isActive ? "page" : undefined}
              className={cn(
                "flex h-9 shrink-0 items-center gap-3 rounded-lg px-[0.65rem] text-left text-[13px]",
                "transition-colors duration-100",
                "focus-visible:ring-ring focus-visible:outline-none focus-visible:ring-2",
                isActive
                  ? "bg-primary/10 text-primary hover:bg-primary/20"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              <Icon className="size-[18px] shrink-0" strokeWidth={1.7} />
              <RailLabel open={open} className="font-medium">
                {d.label}
              </RailLabel>
            </button>
          );
        })}
      </nav>

      {/* The captions only exist to answer "what is this destination for", so
          they render only when the rail is open enough to hold them. Unlike the
          labels they are not an accessible name for anything, so removing them
          when collapsed costs nothing. */}
      {open ? (
        <p className="text-muted-foreground/70 px-4 pb-1 text-[11px] leading-relaxed">
          {DESTINATIONS.find((d) => d.id === active)?.caption}.
        </p>
      ) : null}

      <div className="border-sidebar-border mt-auto shrink-0 border-t py-1">
        <ConnectionRow reach={reach} open={open} />
      </div>
    </aside>
  );
}
