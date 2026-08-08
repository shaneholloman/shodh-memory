import { useMemo } from "react";
import { MapPin, Ruler, Unlink, type LucideIcon } from "lucide-react";
import { ApiError, NetworkError, type Reachability } from "@/lib/api";
import { useCorpus } from "@/lib/api/corpus";
import { useSession } from "@/stores/session";
import { EmptyState } from "@/components/ui/empty-state";
import { InfoHint } from "@/components/ui/info-hint";
import { Meta, Stat } from "@/components/ui/meta";
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
 * row, naming the numbers being compared. Nothing here says "anomalous" and
 * leaves it there.
 *
 * WHERE THE COMPARISON IS SPLIT, AND WHY. Every measure is a comparison, and a
 * comparison has two halves: the baseline, which is the same for every row in a
 * section, and the memory's own magnitude, which is not. Stating both on every
 * row produced the exact wall this screen was rebuilt to remove — four rows
 * reading "260 km / 259 km / 31 km / 18 km from the middle of the cluster the
 * other 35 placed memories form, which normally sit 2.4 km out". So the
 * baseline is stated ONCE, as tokens under the section title, and the row
 * carries only what distinguishes it. The evidence is intact; the sentence is
 * gone.
 *
 * THREE REGISTERS, AND THE RULE THAT SEPARATES THEM. `facts` (the baseline
 * numbers) and `caveat` (anything that changes how those numbers should be
 * read) are always on screen. `looksFor` and `detail` — what the lens is for,
 * and how the arithmetic works — sit behind the section's info affordance,
 * because they are read once and then known. A caveat never goes behind it: the
 * published finding on info tips is that most people never open them, and a
 * limitation nobody reads makes the screen look like it claimed more than it
 * did.
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
  /** What this looks for, before any data — true even when there is none. Read
   *  once and then known, so it lives behind the section's info affordance
   *  rather than as a permanent line above every list. */
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
 * exists to fix. Every row states its own magnitude instead, in the units of
 * its own measure, where it can be checked against the memory above it.
 *
 * The magnitude leads and is set in the mono face, so a column of them is
 * scannable without reading a single label — the same argument that puts
 * numerics in their own right-aligned column in a dense table.
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
        "border-border w-full border-b px-4 py-2.5 text-left transition-colors duration-100",
        "focus-visible:ring-ring focus-visible:-outline-offset-2 focus-visible:ring-2 focus-visible:outline-none",
        selected ? "bg-primary/10" : "hover:bg-accent/60",
      )}
    >
      <p
        className={cn(
          "line-clamp-2 text-[13px] leading-relaxed",
          selected ? "text-foreground" : "text-foreground/90",
        )}
      >
        {finding.content}
      </p>

      {/* The evidence, under the memory it judged. Two tokens, not a sentence:
          what this one measured, and what that measurement is of. The baseline
          it is measured against is the section's, stated once above. */}
      <Meta className="mt-1">
        <Stat value={finding.value} />
        <span>{finding.against}</span>
      </Meta>
    </button>
  );
}

/**
 * A limitation on what the section above could conclude.
 *
 * Marked, indented and always rendered — never folded into the info panel. It
 * is the one thing on this screen that is neither a finding nor an explanation:
 * it says the measure could not reach as far as the reader would assume, and a
 * reader who misses it credits the screen with a stronger result than it has.
 */
function Caveat({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-muted-foreground/80 border-warn/40 mt-2 border-l pl-2.5 text-[11px] leading-relaxed">
      {children}
    </p>
  );
}

function LensSection({ lens }: { lens: Lens }) {
  const Icon = lens.icon;
  const result = lens.result;
  const count = result.state === "findings" ? result.findings.length : 0;

  return (
    <section>
      {/* Title, count, and the affordance that holds everything this section
          used to say in prose — all on the one sticky row, so scrolling past a
          section never loses the ability to ask what it was measuring. */}
      <div className="border-border bg-muted/50 sticky top-0 z-10 flex items-center gap-2 border-b px-4 py-1.5 backdrop-blur-sm">
        <Icon aria-hidden="true" className="text-muted-foreground size-3" strokeWidth={1.8} />
        <span className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
          {lens.title}
        </span>
        {count > 0 ? (
          <span className="text-muted-foreground/60 mono text-[10px]">{count}</span>
        ) : null}
        <InfoHint label={lens.title}>
          <span className="block">{lens.looksFor}</span>
          {result.detail ? (
            <span className="text-muted-foreground mt-1.5 block">{result.detail}</span>
          ) : null}
          {count > 1 ? (
            <span className="text-muted-foreground mt-1.5 block">
              Ordered furthest from the baseline first.
            </span>
          ) : null}
        </InfoHint>
        {/* The section's verdict, right, where the eye finishes the row —
            global identity left, contextual state right, as a status bar
            does. "Clear" and "not enough data" are different claims and each
            gets its own word; neither is ever a blank space. */}
        <span className="text-muted-foreground/60 ml-auto text-[11px]">
          {result.state === "clear"
            ? "Nothing flagged"
            : result.state === "insufficient"
              ? "No baseline"
              : null}
        </span>
      </div>

      <div className="border-border border-b px-4 py-2">
        {/* The baseline: where this lens drew its line on this corpus, as
            numbers. This is the half of every row's comparison that does not
            change, and stating it here is what lets each row below be two
            tokens instead of a sentence. */}
        <Meta>
          {result.facts.map((f) => (
            <span key={f}>{f}</span>
          ))}
        </Meta>
        {result.state === "insufficient" ? <Caveat>{result.reason}</Caveat> : null}
        {result.state !== "insufficient" && result.caveat ? (
          <Caveat>{result.caveat}</Caveat>
        ) : null}
      </div>

      {result.state === "findings"
        ? result.findings.map((f) => (
            <FindingRow key={`${lens.id}-${f.memoryId}-${f.value}`} finding={f} />
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
        body="These measures read this profile's memory, which needs the server running."
      />
    );
  }

  if (profile === null) {
    return (
      <EmptyState
        size="page"
        title="No profile to measure"
        body="A baseline is built from one profile's memories."
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
        body="Nothing to compare against."
        more="Every measure here compares a memory with the others in the same profile, so a baseline needs memories to be built from. These appear as soon as anything is stored."
      />
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-2xl pb-16">
        {/* The header a stranger reads first, and a returning reader skips.
            It is a count, not a claim about the screen: the destination is
            already named in the top bar, and repeating "what stands out in
            this profile's memory" underneath it was two lines of sentence
            saying what one number says.

            The one thing that is NOT obvious from looking — that these
            findings are arithmetic over the store's own contents rather than a
            judgement fetched from somewhere else — stays on screen as a chip,
            because it is the product's actual claim and a claim behind an icon
            is a claim nobody reads. The icon holds only how it works. */}
        <header className="border-border flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b px-4 py-3">
          <Meta className="text-[12px]">
            <Stat value={flagged} label="flagged" />
            <Stat value={memories.length} label="memories" />
            <Stat value={lenses.length} label="measures" />
          </Meta>
          {/* Not "No model": the seat's egress badge already occupies that
              exact phrase in the corner of every screen, and two unrelated
              claims wearing one wording is worse than either being longer. */}
          <span className="border-border text-muted-foreground ml-auto flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px]">
            Arithmetic, not a model
            <InfoHint label="how these are found">
              Every finding on this screen is arithmetic over the memories already stored here — a
              median centre and a robust spread, unit-matched number comparison, and term overlap.
              No model is called, here or on the server, and nothing is inferred that the corpus
              does not already state.
            </InfoHint>
          </span>
        </header>

        {lenses.map((lens) => (
          <LensSection key={lens.id} lens={lens} />
        ))}
      </div>
    </ScrollArea>
  );
}
