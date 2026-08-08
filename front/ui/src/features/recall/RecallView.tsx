import { useMemo } from "react";
import { cn } from "@/lib/utils";
import { ApiError, NetworkError, type Reachability } from "@/lib/api";
import { corpusToRecallMemory, useCorpus, type CorpusMemory } from "@/lib/api/corpus";
import { EmptyState } from "@/components/ui/empty-state";
import { useSession } from "@/stores/session";
import { ResultList, ResultListSkeleton } from "./ResultList";
import { GraphStage } from "./GraphStage";
import { useRecall } from "./useRecall";
import { corpusCues } from "./cues";

/**
 * Recall — the search surface, and the entry point to both chains.
 *
 * The query is submit-driven rather than live. Recall runs a multi-leg
 * retrieval (vector + BM25 + graph, then fusion and re-ranking); firing it on
 * every keystroke would spend that repeatedly to answer prefixes nobody asked
 * about. A search field that acts on Enter is also what the control already
 * promises.
 *
 * Both states of the result column are headed, and the heading is the column's
 * only chance to say what the rows underneath it ARE. Before a query they are
 * the corpus, newest first, and the head offers the first move. After one they
 * are an answer, and the head says how much was searched, how long it took, and
 * the one thing about this product that a competing screenful of results cannot
 * claim: none of it was written.
 */

/** How many of the newest memories the pre-query listing shows. Enough to make
 *  the column obviously scrollable and to convey what kind of thing is stored;
 *  not so many that the browser lays out the whole corpus to say it. */
const RECENT_LIMIT = 50;

/**
 * The pre-query head: what is here, and what to do first.
 *
 * The cues are the first action. They are not suggestions in the marketing
 * sense — each one is a tag the extraction pipeline wrote against these
 * memories, shown verbatim, and clicking it runs it as the query (see cues.ts
 * for why they can be trusted on a corpus nobody has seen). On a corpus whose
 * tags are all lower-case debris this renders no row at all, which is the
 * correct outcome: an empty row is honest, an invented cue is not.
 */
function CorpusHead({ memories, total }: { memories: CorpusMemory[] | undefined; total: number }) {
  const setActiveQuery = useSession((s) => s.setActiveQuery);
  const cues = useMemo(() => corpusCues(memories), [memories]);

  return (
    <div className="border-border shrink-0 border-b px-4 py-2.5">
      <p className="text-muted-foreground text-[11px] leading-relaxed">
        {total} {total === 1 ? "memory" : "memories"} here, newest first. Search to rank
        them by relevance.
      </p>

      {cues.length > 0 ? (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          {/* Named, because four bare words under a sentence read as tags on the
              screen rather than as things to press. */}
          <span className="text-muted-foreground/60 mr-0.5 text-[11px]">Mentioned most</span>
          {cues.map((cue) => (
            <button
              key={cue.text}
              type="button"
              onClick={() => setActiveQuery(cue.text)}
              // The visible label is a word and a number, which announces as
              // "Seagirt 10" and says neither what pressing it does nor what
              // the number counts. The heading above supplies both visually and
              // is not associated with the control, so the control says it.
              aria-label={`Search for ${cue.text} — mentioned in ${cue.count} ${
                cue.count === 1 ? "memory" : "memories"
              }`}
              className={cn(
                "border-border text-foreground/90 hover:border-ring/50 hover:bg-accent/60",
                "focus-visible:ring-ring flex items-baseline gap-1 rounded-full border px-2 py-0.5",
                "text-[11px] transition-colors duration-100 focus-visible:ring-2 focus-visible:outline-none",
              )}
            >
              {cue.text}
              {/* The count is why these four and not four others. Without it the
                  row is a taste; with it, it is a reading of the corpus. */}
              <span className="text-muted-foreground/60 mono text-[10px]">{cue.count}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/**
 * The post-query head.
 *
 * Two facts and one claim, and every part of it is read off the response rather
 * than asserted:
 *
 *  - the count of what surfaced against the count of what was there to search;
 *  - `retrieval_stats.retrieval_time_us`, which the server measures and returns
 *    on every debug query (types.ts cites the struct). It is SERVER-SIDE
 *    RETRIEVAL time and is labelled as such — it excludes the network hop and
 *    everything the browser does, so calling it "response time" would overclaim
 *    a number that is not that. When the field is absent the clause is absent;
 *    there is no fallback estimate, because an estimate presented in the same
 *    typeface as a measurement is worse than silence;
 *  - "nothing here was written by a model", which is the product's actual edge
 *    and is checkable on this screen: every row is a stored memory rendered
 *    verbatim, and the Inspector traces each one to what recorded it. The
 *    stronger-sounding claim — that no model runs at all — is NOT made, because
 *    it is false: retrieval embeds the query with a local encoder
 *    (src/embeddings/minilm.rs, and `embedding_time_us` is a line item in the
 *    server's own timing). Generation is what is absent here, and generation is
 *    what the claim says.
 */
function AnswerHead({
  shown,
  total,
  retrievalUs,
}: {
  shown: number;
  total: number;
  retrievalUs: number | undefined;
}) {
  return (
    <div className="border-border shrink-0 border-b px-4 py-2.5">
      <p className="text-muted-foreground text-[11px]">
        {/* "Top", not a bare count: `shown` is capped by the request limit
            (lib/api/recall.ts sends 25), so "25 of 74" would read as "25
            matched" when it means "the 25 best of what matched". "Top 5 of 74"
            stays exact when fewer come back. */}
        Top {shown} of {total} {total === 1 ? "memory" : "memories"}
        {retrievalUs !== undefined ? (
          <span title="Time the server spent retrieving and ranking. Excludes the network round trip.">
            {" · retrieved in "}
            <span className="mono">{Math.round(retrievalUs / 1000)} ms</span>
          </span>
        ) : null}
      </p>
      <p className="text-muted-foreground/60 mt-1 text-[11px] leading-relaxed">
        Nothing here was written by a model — these are stored memories, ranked.
      </p>
    </div>
  );
}

function ResultPane({ reach }: { reach: Reachability }) {
  // The query itself lives in `useRecall` — the graph stage renders the same
  // result set and must not issue a second retrieval to get it.
  const { data, error, isFetching, profile, query } = useRecall(reach);
  const corpus = useCorpus(reach);

  // The pre-query listing: newest first. The corpus endpoint already returns
  // newest-first, but sorting here keeps the promise in the heading true even
  // if that ordering ever changes server-side.
  const recent = useMemo(
    () =>
      [...(corpus.data?.memories ?? [])]
        .sort((a, b) => b.created_at.localeCompare(a.created_at))
        .slice(0, RECENT_LIMIT)
        .map(corpusToRecallMemory),
    [corpus.data],
  );

  if (reach.state !== "online") {
    return (
      <EmptyState
        title="Not connected"
        body="Results appear here once the memory server is running."
      />
    );
  }

  if (profile === null) {
    return (
      <EmptyState
        title="No profile to search"
        body="This instance holds no memory yet. Recall needs a profile that already exists — searching would otherwise create an empty one."
      />
    );
  }

  if (!query.trim()) {
    // No query yet: open onto the corpus, newest first, instead of onto an
    // instruction. The list IS the invitation — it shows what there is to
    // search, and selecting a row inspects it like any result.
    if (corpus.isFetching && !corpus.data) return <ResultListSkeleton />;
    if (recent.length === 0) {
      return (
        <EmptyState
          title="Nothing remembered yet"
          body="Once this profile holds memory, the newest entries appear here before any search."
        />
      );
    }
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <CorpusHead
          memories={corpus.data?.memories}
          total={corpus.data?.total ?? recent.length}
        />
        <ResultList memories={recent} />
      </div>
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
          : "Something went wrong running this query.";
    return <EmptyState title="Recall failed" body={detail} />;
  }

  if (isFetching && !data) {
    return <ResultListSkeleton />;
  }

  if (data && data.memories.length === 0) {
    return (
      <EmptyState
        title="Nothing surfaced"
        body="No memory in this profile activated strongly enough for that cue. A shorter, more general cue usually reaches further."
      />
    );
  }

  return data ? (
    <div className="flex min-h-0 flex-1 flex-col">
      <AnswerHead
        shown={data.memories.length}
        total={corpus.data?.total ?? data.memories.length}
        retrievalUs={data.retrieval_stats?.retrieval_time_us}
      />
      {/* The lineage the server returned with this result set, so a row can say
          how many of the others it is causally linked to. Same object the graph
          canvas plots and the Inspector walks — one response, three readings. */}
      <ResultList memories={data.memories} lineage={data.lineage} />
    </div>
  ) : null;
}

export function RecallView({ reach }: { reach: Reachability }) {
  return (
    <div className="flex h-full min-h-0">
      {/* `min(340px,42vw)`, not a bare 340px: `main`'s content box is the
          viewport minus the 56px rail and the Inspector's reserved width
          (see INSPECTOR_OFFSET in App.tsx). Below ~676px a fixed 340px
          column no longer fits next to a fixed-width Inspector — it was
          overflowing past the Inspector's left edge and rendering underneath
          it, invisibly, because `body`'s global `overflow: hidden` clips
          rather than scrolls; GraphStage was squeezed to 0 width in the same
          collapse. The vw cap is a no-op at desktop widths (340px is
          untouched above ~810px) and shrinks the column smoothly below that
          instead of letting it overflow. */}
      <div className="border-border flex w-[min(340px,42vw)] shrink-0 flex-col border-r">
        <ResultPane reach={reach} />
      </div>
      <GraphStage reach={reach} />
    </div>
  );
}
