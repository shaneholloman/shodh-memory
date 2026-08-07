import { cn } from "@/lib/utils";
import type { RecallMemory } from "@/lib/api";
import { useSession } from "@/stores/session";

/**
 * The result column.
 *
 * A row is the entry point to both chains in WORKFLOWS.md — selecting it opens
 * the Inspector, which is the only object-detail surface in the product. So the
 * row's job is to carry just enough to choose between candidates, and nothing
 * more: the content, when it was recorded, and how strongly it surfaced.
 *
 * The score shown is the server's normalised `score`, not the raw fusion
 * output. It is rendered as a bar rather than a number because the useful
 * question at this stage is "how does this compare to the one above it", not
 * "what is its absolute value" — and a decimal invites false precision about a
 * quantity that is only meaningful relative to the top hit.
 */

function relativeDay(iso: string): string {
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return "";
  const days = Math.floor((Date.now() - then.getTime()) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days}d ago`;
  if (days < 365) return `${Math.floor(days / 30)}mo ago`;
  return `${Math.floor(days / 365)}y ago`;
}

function ResultRow({ memory }: { memory: RecallMemory }) {
  const selected = useSession((s) => s.selectedMemoryId === memory.id);
  const select = useSession((s) => s.select);

  return (
    <button
      type="button"
      onClick={() => select(memory.id)}
      aria-current={selected ? "true" : undefined}
      className={cn(
        "border-border w-full border-b px-4 py-3 text-left transition-colors duration-100",
        "focus-visible:ring-ring focus-visible:-outline-offset-2 focus-visible:ring-2 focus-visible:outline-none",
        selected ? "bg-primary/10" : "hover:bg-accent/60",
      )}
    >
      <p
        className={cn(
          "line-clamp-3 text-[13px] leading-relaxed",
          selected ? "text-foreground" : "text-foreground/90",
        )}
      >
        {memory.experience.content}
      </p>

      <div className="mt-2 flex items-center gap-2">
        {/* Relative strength against the top hit, which is what the server's
            normalisation already expresses. */}
        <span className="bg-border h-[3px] w-16 shrink-0 overflow-hidden rounded-full">
          <span
            className={cn("block h-full rounded-full", selected ? "bg-primary" : "bg-muted-foreground/60")}
            style={{ width: `${Math.max(2, Math.min(100, memory.score * 100))}%` }}
          />
        </span>
        <span className="text-muted-foreground mono text-[10px]">
          {relativeDay(memory.created_at)}
        </span>
        {memory.experience.geo_location ? (
          <span className="text-muted-foreground mono text-[10px]" title="Has coordinates">
            {memory.experience.geo_location[0].toFixed(2)},
            {memory.experience.geo_location[1].toFixed(2)}
          </span>
        ) : null}
      </div>
    </button>
  );
}

export function ResultList({ memories }: { memories: RecallMemory[] }) {
  return (
    <div role="list" className="min-h-0 flex-1 overflow-y-auto">
      {memories.map((m) => (
        <div role="listitem" key={m.id}>
          <ResultRow memory={m} />
        </div>
      ))}
    </div>
  );
}
