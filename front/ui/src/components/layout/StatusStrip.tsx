import { cn } from "@/lib/utils";
import { isHumanProfile, type Reachability } from "@/lib/api";

/**
 * Connection state, said once for the whole product.
 *
 * Gridline's segmented strip, carrying the live state this app actually has. It
 * replaces three separate statements of the same fact — a card in the middle of
 * the stage, a row at the foot of the rail, and a line in the Inspector — which
 * between them made an ordinary not-started-yet state read as broken.
 *
 * A state and its remedy travel together. "Offline" alone sends people to check
 * their network when the server simply is not running.
 *
 * The fourth state is the one that only exists because the server tells us the
 * profile list: reachable, authorized, and holding no profiles at all. That is
 * a fresh install, and it is not an error — but it is also not usable, because
 * `get_user_memory` provisions a store for any id it is handed, so inventing a
 * profile to search against would silently create one.
 */
export function StatusStrip({ reach }: { reach: Reachability }) {
  const { tone, state, remedy } =
    reach.state === "online"
      ? reach.profiles.filter(isHumanProfile).length === 0
        ? {
            tone: "text-warn",
            state: "No profiles",
            remedy: "Connected, but this instance holds no memory yet",
          }
        : {
            tone: "text-[var(--live)]",
            state: "Connected",
            remedy: (() => {
              const count = reach.profiles.filter(isHumanProfile).length;
              return `${count} profile${count === 1 ? "" : "s"}`;
            })(),
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
