import { useQuery } from "@tanstack/react-query";
import { recall, ApiError, NetworkError, type Reachability } from "@/lib/api";
import { useSession } from "@/stores/session";
import { EmptyState } from "@/components/ui/empty-state";
import { ResultList, ResultListSkeleton } from "./ResultList";
import { GraphStage } from "./GraphStage";

/**
 * Recall — the search surface, and the entry point to both chains.
 *
 * The query is submit-driven rather than live. Recall runs a multi-leg
 * retrieval (vector + BM25 + graph, then fusion and re-ranking); firing it on
 * every keystroke would spend that repeatedly to answer prefixes nobody asked
 * about. A search field that acts on Enter is also what the control already
 * promises.
 */

function ResultPane({ reach }: { reach: Reachability }) {
  const profile = useSession((s) => s.profile);
  const query = useSession((s) => s.activeQuery);

  const enabled = reach.state === "online" && profile !== null && query.trim().length > 0;

  const { data, error, isFetching } = useQuery({
    // The profile is part of the key, not just the payload: switching profile
    // is switching corpus, and showing the previous one's hits under a new
    // profile would be a correctness bug, not a stale-cache annoyance.
    queryKey: ["recall", profile, query],
    queryFn: ({ signal }) => recall({ user_id: profile!, query }, signal),
    enabled,
  });

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
    return (
      <EmptyState
        title="Search memory"
        body="A few words are enough. Recall starts from meaning, not from matching the words that were stored."
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

  return data ? <ResultList memories={data.memories} /> : null;
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
      <GraphStage />
    </div>
  );
}
