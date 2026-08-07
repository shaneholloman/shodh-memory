import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import type { Reachability } from "@/lib/api";
import { StatusStrip } from "./StatusStrip";

/** Collapsed rail width, as the offset every fixed element reserves. The rail's
 *  *expanded* width is deliberately never reserved: doing so would make the
 *  expansion push content, which is the one thing it must not do. */
export const RAIL_OFFSET = "pl-14";

export function TopBar({
  title,
  reach,
  children,
}: {
  title: string;
  reach: Reachability;
  children?: ReactNode;
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
