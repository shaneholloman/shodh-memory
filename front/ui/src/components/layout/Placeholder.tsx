import type { Reachability } from "@/lib/api";
import { EmptyState } from "@/components/ui/empty-state";
import { DESTINATIONS, type DestinationId } from "./Sidebar";

/** A destination that has no content yet says so plainly, in its own terms. */
export function Placeholder({ id, reach }: { id: DestinationId; reach: Reachability }) {
  const dest = DESTINATIONS.find((d) => d.id === id);
  return (
    <EmptyState
      size="page"
      title={dest?.label ?? ""}
      body={`${dest?.caption}. ${
        reach.state === "online" ? "Nothing to show yet." : "Start the memory server to see it."
      }`}
    />
  );
}
