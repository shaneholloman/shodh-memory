import { Search, TriangleAlert, ListChecks } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Reachability } from "@/lib/api";
import shodhMark from "@/assets/shodh-mark.png";

/**
 * Primary navigation.
 *
 * Moved out of a horizontal tab strip into a left rail, which is where a
 * product of this shape puts destinations. The strip also carried 1/2/3 key
 * hints that existed only because the terminal UI has a ViewMode enum — an
 * internal reason showing through the product.
 *
 * Names are what the thing is, not what the subsystem is called:
 *   Recall     — the search surface (was "Live")
 *   Anomalies  — deviations from this user's own baseline (was unqualified)
 *   Tasks      — the todo list (was "Work", which named nothing)
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

function ConnectionRow({ reach }: { reach: Reachability }) {
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
    <div className="text-muted-foreground flex items-center gap-2 px-2 py-1.5 text-[11px]">
      <span className={cn("size-1.5 shrink-0 rounded-full", tone)} />
      <span className="truncate">{label}</span>
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
  return (
    <nav className="border-border bg-card flex w-[212px] flex-col border-r">
      <div className="flex items-center gap-2 px-3 py-3">
        <img
          src={shodhMark}
          alt=""
          aria-hidden="true"
          className="size-7 shrink-0 object-contain"
        />
        <div className="min-w-0 leading-tight">
          <div className="truncate text-[13px] font-semibold tracking-tight">
            shodh
          </div>
          <div className="text-muted-foreground truncate text-[11px]">
            {identity}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-0.5 px-2 py-1">
        {DESTINATIONS.map((d) => {
          const Icon = d.icon;
          const isActive = d.id === active;
          return (
            <button
              key={d.id}
              type="button"
              onClick={() => onNavigate(d.id)}
              aria-current={isActive ? "page" : undefined}
              title={d.caption}
              className={cn(
                "flex items-center gap-2 rounded-md px-2 py-1.5 text-left text-[13px]",
                "transition-colors duration-100",
                "focus-visible:ring-ring focus-visible:outline-none focus-visible:ring-2",
                isActive
                  ? "bg-accent text-accent-foreground font-medium"
                  : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
              )}
            >
              <Icon className="size-4 shrink-0" />
              <span className="truncate">{d.label}</span>
            </button>
          );
        })}
      </div>

      <div className="border-border mt-auto border-t p-1">
        <ConnectionRow reach={reach} />
      </div>
    </nav>
  );
}
