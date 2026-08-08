import { useMemo } from "react";
import type { Reachability } from "@/lib/api";
import { corpusToRecallMemory, useCorpus } from "@/lib/api/corpus";
import { EmptyState } from "@/components/ui/empty-state";
import { useRecall } from "@/features/recall/useRecall";
import { useMemoryTypes } from "@/features/recall/GraphCanvas";
import { GeoMap } from "./GeoMap";

/**
 * Geo — everywhere this profile's memory happened, with the current recall
 * result highlighted on top of it.
 *
 * The map is populated the moment the destination opens: every located memory
 * in the corpus is drawn as a quiet context point, because "where do I have
 * memory?" is a question with an answer before any search is typed. Running a
 * search does not swap the map out — it turns the matching points up and the
 * rest down, so a result is always seen against the corpus it came from.
 *
 * Both data sources are shared cache entries (useCorpus, useRecall): this view
 * issues no retrieval of its own.
 */

export function GeoView({ reach }: { reach: Reachability }) {
  const corpus = useCorpus(reach);
  const { data, error, isFetching, profile, query } = useRecall(reach);

  const hasQuery = query.trim().length > 0;
  const results = useMemo(() => (hasQuery ? (data?.memories ?? []) : []), [hasQuery, data]);

  // The plotted set: recall results first (they carry scores and therefore
  // size), then every located corpus memory that is not already a result.
  const { plotted, dimmed } = useMemo(() => {
    const resultIds = new Set(results.map((m) => m.id));
    const context = (corpus.data?.memories ?? [])
      .filter((m) => m.geo_location)
      .filter((m) => !resultIds.has(m.id))
      .map(corpusToRecallMemory);
    if (!hasQuery || results.length === 0) {
      // No active answer: the corpus IS the map. Nothing is dimmed — these
      // points are not losing to anything.
      return { plotted: context, dimmed: undefined };
    }
    return {
      plotted: [...results, ...context],
      dimmed: new Set(context.map((m) => m.id)),
    };
  }, [corpus.data, results, hasQuery]);

  const types = useMemoryTypes(plotted);
  const located = plotted.filter((m) => m.experience.geo_location);
  const matched = results.filter((m) => m.experience.geo_location);

  if (reach.state !== "online") {
    return (
      <EmptyState
        size="page"
        title="Not connected"
        body="The map draws from memory, which needs the memory server running."
      />
    );
  }

  if (profile === null) {
    return (
      <EmptyState
        size="page"
        title="No profile selected"
        body="The map shows where a profile's memory happened, so it needs a profile that already exists."
      />
    );
  }

  if (corpus.isFetching && !corpus.data) {
    return <EmptyState size="page" title="Loading corpus" body="Placing every located memory." />;
  }

  if (located.length === 0) {
    return (
      // Factual, not apologetic: the reason is a property of the data.
      <EmptyState
        size="page"
        title="No coordinates in this corpus"
        body="A memory only carries coordinates when whatever wrote it supplied them — imported corpora like GDELT do, session captures do not. Once one does, it appears here without any search."
      />
    );
  }

  return (
    // `h-full`, not `flex-1` — this view is a direct child of `main`, which is
    // not a flex container (see the history of this comment in git blame).
    <section className="relative h-full min-h-0 min-w-0 overflow-hidden">
      <div aria-hidden="true" className="graticule pointer-events-none absolute inset-0" />

      <GeoMap memories={plotted} types={types} dimmed={dimmed} />

      <div className="pointer-events-none absolute top-3 left-4 z-10 flex flex-col gap-0.5">
        <span className="text-muted-foreground text-[12px]">
          {hasQuery ? "Where this answer happened" : "Where this profile's memory happened"}
        </span>
        {/* Said only before a query, because it is the one thing this map does
            that is not visible from looking at it: searching does not swap the
            points out, it turns the matching ones up. Without the line, a map
            that stays put after a search reads as a map that ignored it. */}
        {!hasQuery ? (
          <span className="text-muted-foreground/60 text-[11px]">
            Search to raise the points that match
          </span>
        ) : null}
      </div>

      <div
        className="pointer-events-none absolute inset-x-4 bottom-3 z-10 flex flex-wrap items-center justify-between gap-x-6 gap-y-2"
        style={{ paddingRight: "var(--overlay-dock-inset, 0px)" }}
      >
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
          {types.map((t, i) => (
            <span key={t} className="text-muted-foreground flex items-center gap-1.5 text-[11px]">
              <span
                className="size-2 rounded-full"
                style={{ background: `var(--chart-${(i % 5) + 1})` }}
              />
              {t}
            </span>
          ))}
        </div>
        <span className="text-muted-foreground/70 text-[11px]">
          {hasQuery
            ? error
              ? "That search did not complete — showing the corpus"
              : isFetching && !data
                ? "Searching…"
                : `${matched.length} matched of ${located.length} placed`
            : `${located.length} located ${located.length === 1 ? "memory" : "memories"}`}
          {" · scroll to zoom · drag to pan · click a point to inspect"}
        </span>
      </div>
    </section>
  );
}
