import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Reachability } from "@/lib/api";

/**
 * Top bar. Linear keeps this thin, quiet, and hairline-separated: it is
 * orientation, not a place to put weight. The one saturated element permitted
 * here is the status dot, because connection state is the only thing on this
 * bar a person needs to notice without looking for it.
 */

function StatusDot({ reach }: { reach: Reachability }) {
  const tone =
    reach.state === "online"
      ? "bg-[var(--live)]"
      : reach.state === "unauthorized"
        ? "bg-destructive"
        : "bg-muted-foreground/50";

  const label =
    reach.state === "online"
      ? "Connected"
      : reach.state === "unauthorized"
        ? "Key rejected"
        : "Offline";

  return (
    <div className="flex items-center gap-2">
      <span className={cn("size-1.5 rounded-full", tone)} />
      <span className="text-muted-foreground text-[12px]">{label}</span>
    </div>
  );
}

/** A claim about how this product runs. Quiet by design — these are facts, not badges. */
function Claim({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-muted-foreground border-border rounded-md border px-2 py-1 text-[11px] font-medium">
      {children}
    </span>
  );
}

export function TopBar({
  reach,
  identity,
}: {
  reach: Reachability;
  identity: string;
}) {
  return (
    <header className="border-border flex h-12 flex-shrink-0 items-center gap-4 border-b px-4">
      <div className="flex items-baseline gap-2">
        <span className="text-[15px] font-semibold tracking-tight">shodh</span>
        <span className="text-muted-foreground text-[12px]">
          Associative Memory
        </span>
      </div>

      <StatusDot reach={reach} />

      <div className="ml-auto flex items-center gap-2">
        <button
          type="button"
          className={cn(
            "border-border bg-card hover:bg-accent flex items-center gap-2",
            "rounded-md border px-2.5 py-1.5 text-[12px] font-medium",
            "transition-colors duration-100",
            "focus-visible:ring-ring focus-visible:outline-none focus-visible:ring-2",
          )}
        >
          <span>{identity}</span>
          <ChevronDown className="text-muted-foreground size-3.5" />
        </button>

        <div className="flex items-center gap-1.5">
          <Claim>LLM-free</Claim>
          <Claim>Deterministic</Claim>
          <Claim>On-device</Claim>
        </div>
      </div>
    </header>
  );
}
