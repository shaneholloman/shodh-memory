import { useMemo } from "react";
import { MapPin, Ruler, Unlink, type LucideIcon } from "lucide-react";
import { ApiError, NetworkError, type Reachability } from "@/lib/api";
import { useCorpus } from "@/lib/api/corpus";
import { useSession } from "@/stores/session";
import { EmptyState } from "@/components/ui/empty-state";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import {
  isolatedMemories,
  offPatternLocations,
  quantityOutliers,
  type Finding,
  type LensResult,
} from "./measures";

/**
 * Anomalies — what deviates from this profile's own baseline.
 *
 * The destination exists to answer three questions a reader has in front of
 * any result: what am I looking at, why was this one chosen, and what did the
 * choosing. So every finding carries the measure that produced it, in the same
 * row, in words that name the numbers being compared. Nothing here says
 * "anomalous" and leaves it there.
 *
 * The measures are in measures.ts and are pure arithmetic — a median centre and
 * a median absolute deviation over great-circle distance, unit-matched number
 * comparison, and term overlap. No model is called, here or on the server, and
 * that is the claim the header makes: the store finds these in its own data.
 *
 * NO REQUEST OF ITS OWN. It reads `useCorpus`, the same single react-query
 * entry Recall and Geo already populate (lib/api/corpus.ts), so arriving here
 * from either is instant and arriving here first costs one listing the rest of
 * the app then reuses.
 *
 * A section that cannot reach a conclusion says why, with the shortfall in
 * numbers, instead of rendering an empty box that reads as "clean". "Not enough
 * placed memories to have a baseline" and "no memory is out of place" are
 * different findings and this screen never lets one impersonate the other.
 */

interface Lens {
  id: string;
  icon: LucideIcon;
  title: string;
  /** What this looks for, before any data — true even when there is none. */
  looksFor: string;
  result: LensResult;
}

/**
 * One flagged memory.
 *
 * Selecting sets the one global selection (`useSession().select`), exactly as
 * a recall result row does, so an anomaly is the same kind of object as any
 * other memory in the product and hands off to the same Inspector. The row also
 * shows its own selected state, so the click reads as having landed even before
 * the detail pane resolves.
 *
 * SEVERITY IS POSITION, AND ONLY POSITION. Rows are ordered furthest-first and
 * carry no severity mark. Not a colour, because hue in this product means
 * memory type and a second categorical palette on the same surface would put
 * two encodings on one channel. Not a bar either, and that one was tried and
 * removed: the underlying magnitudes are unbounded, so on a real corpus where
 * one memory sits 126 robust scales out and the next sits 14, a bar scaled to
 * the largest drew the second as an empty stub — it made a genuine finding look
 * like a rounding error, which is the exact clarity defect this destination
 * exists to fix. Every row states its own magnitude in words instead, in the
 * units of its own measure, where it can be checked against the memory above
 * it.
 */
function FindingRow({ finding }: { finding: Finding }) {
  const selected = useSession((s) => s.selectedMemoryId === finding.memoryId);
  const select = useSession((s) => s.select);

  return (
    <button
      type="button"
      onClick={() => select(finding.memoryId)}
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
        {finding.content}
      </p>

      {/* The measure sits under the memory it judged, indented by the same
          rule the section's own basis line uses, so "this is the reasoning
          about the thing above" reads the same way at both levels. */}
      <p className="text-muted-foreground border-border/60 mt-2 border-l pl-3 text-[11px] leading-relaxed">
        {finding.measure}
      </p>
    </button>
  );
}

/**
 * The sentence explaining how a section reached its conclusion.
 *
 * Present in every state, including the ones with no findings, because the
 * mechanism is the point: a reader who learns how the line was drawn can judge
 * an empty section as readily as a full one.
 */
function Basis({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-muted-foreground/70 border-border/60 mt-2 border-l pl-3 text-[11px] leading-relaxed">
      {children}
    </p>
  );
}

function LensSection({ lens }: { lens: Lens }) {
  const Icon = lens.icon;
  const count = lens.result.state === "findings" ? lens.result.findings.length : 0;

  return (
    <section>
      <div className="border-border bg-muted/50 sticky top-0 z-10 flex items-center gap-2 border-b px-4 py-1.5 backdrop-blur-sm">
        <Icon aria-hidden="true" className="text-muted-foreground size-3" strokeWidth={1.8} />
        <span className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
          {lens.title}
        </span>
        {count > 0 ? (
          <span className="text-muted-foreground/60 mono text-[10px]">{count}</span>
        ) : null}
      </div>

      <div className="border-border border-b px-4 py-3">
        <p className="text-foreground/80 text-[12px] leading-relaxed">{lens.looksFor}</p>
        {lens.result.state === "insufficient" ? (
          <Basis>{lens.result.reason}</Basis>
        ) : (
          <Basis>
            {lens.result.basis}
            {/* Order is the only severity encoding on this screen, so it is
                stated rather than left to be inferred. */}
            {count > 1 ? " Furthest from the baseline first." : ""}
          </Basis>
        )}
      </div>

      {lens.result.state === "findings"
        ? lens.result.findings.map((f) => (
            <FindingRow key={`${lens.id}-${f.memoryId}-${f.measure}`} finding={f} />
          ))
        : null}
    </section>
  );
}

function LensSkeleton() {
  return (
    <section>
      <div className="border-border border-b px-4 py-1.5">
        <Skeleton className="h-3 w-32" />
      </div>
      <div className="border-border border-b px-4 py-3">
        <Skeleton className="h-3 w-[70%]" />
        <Skeleton className="mt-2 h-2.5 w-[90%]" />
      </div>
      <div className="border-border border-b px-4 py-3">
        <Skeleton className="h-[13px] w-full" />
        <Skeleton className="mt-1.5 h-[13px] w-[62%]" />
        <Skeleton className="mt-2.5 h-2.5 w-[80%]" />
      </div>
    </section>
  );
}

export function AnomaliesView({ reach }: { reach: Reachability }) {
  const { data, error, isFetching, profile } = useCorpus(reach);
  const memories = useMemo(() => data?.memories ?? [], [data]);

  // Three passes over one already-fetched list. Memoised on the corpus rather
  // than on the query object so a background refetch that returns identical
  // data does not recompute — and, more to the point, does not reorder rows
  // under a reader's cursor.
  const lenses = useMemo<Lens[]>(
    () => [
      {
        id: "location",
        icon: MapPin,
        title: "Out of place",
        looksFor:
          "Memories whose coordinates sit outside the cluster the rest of this profile's placed memories form.",
        result: offPatternLocations(memories),
      },
      {
        id: "quantity",
        icon: Ruler,
        title: "Numbers that do not fit",
        looksFor:
          "Figures written in the same unit that disagree — inside one memory, or against the range every other memory in that unit establishes.",
        result: quantityOutliers(memories),
      },
      {
        id: "isolation",
        icon: Unlink,
        title: "Connected to nothing",
        looksFor:
          "Memories that name nothing any other memory here names, so no path in the graph reaches them.",
        result: isolatedMemories(memories),
      },
    ],
    [memories],
  );

  const flagged = useMemo(() => {
    const ids = new Set<string>();
    for (const lens of lenses) {
      if (lens.result.state !== "findings") continue;
      for (const f of lens.result.findings) ids.add(f.memoryId);
    }
    return ids.size;
  }, [lenses]);

  if (reach.state !== "online") {
    return (
      <EmptyState
        size="page"
        title="Not connected"
        body="Deviations are measured against this profile's own memory, which needs the memory server running."
      />
    );
  }

  if (profile === null) {
    return (
      <EmptyState
        size="page"
        title="No profile to measure"
        body="A baseline is built from one profile's memories, so there has to be a profile before anything can deviate from it."
      />
    );
  }

  if (error) {
    const detail =
      error instanceof ApiError
        ? error.isAuthFailure
          ? "The server rejected this key."
          : `The server answered ${error.status}.`
        : error instanceof NetworkError
          ? "The server stopped responding mid-request."
          : "Something went wrong loading this profile's memories.";
    return <EmptyState size="page" title="Could not read the corpus" body={detail} />;
  }

  if (isFetching && !data) {
    return (
      <div className="mx-auto h-full w-full max-w-2xl">
        <div className="border-border border-b px-4 py-4">
          <Skeleton className="h-3.5 w-[60%]" />
          <Skeleton className="mt-2 h-2.5 w-[85%]" />
        </div>
        <LensSkeleton />
        <LensSkeleton />
      </div>
    );
  }

  if (memories.length === 0) {
    return (
      <EmptyState
        size="page"
        title="Nothing remembered yet"
        body="Every measure on this screen compares a memory against the others in the same profile. With none stored, there is no baseline and nothing to compare."
      />
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-2xl pb-16">
        <header className="border-border border-b px-4 py-4">
          {/* The one line a stranger reads first. It says what the screen is
              and, in the same breath, what makes it unusual: the finding is
              arithmetic over the store's own contents, not a judgement fetched
              from somewhere else. */}
          <h1 className="text-[14px] font-medium tracking-tight">
            What stands out in this profile's memory — measured against the rest of it, by
            arithmetic rather than by a model.
          </h1>
          <p className="text-muted-foreground mt-1.5 text-[12px] leading-relaxed">
            Three measures run over the {memories.length} memories stored here, each comparing a
            memory with its own corpus and naming the comparison it made.{" "}
            {flagged > 0
              ? `${flagged} ${flagged === 1 ? "memory is" : "memories are"} flagged below; select one to inspect it.`
              : "Nothing is flagged; each measure states below what it looked at."}
          </p>
        </header>

        {lenses.map((lens) => (
          <LensSection key={lens.id} lens={lens} />
        ))}
      </div>
    </ScrollArea>
  );
}
