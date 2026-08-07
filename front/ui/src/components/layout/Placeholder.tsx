import type { Reachability } from "@/lib/api";
import { DESTINATIONS, type DestinationId } from "./Sidebar";

/** A destination that has no content yet says so plainly, in its own terms. */
export function Placeholder({ id, reach }: { id: DestinationId; reach: Reachability }) {
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
