import { useEffect, useMemo, useState } from "react";
import { ChevronRight } from "lucide-react";
import type { Reachability } from "@/lib/api";
import { EmptyState } from "@/components/ui/empty-state";
import { useSession } from "@/stores/session";
import { useUniverse } from "./useUniverse";
import { EntityCanvas, levelFor, type Level } from "./EntityCanvas";
import { entityTypeToken, NAMED_ENTITY_TYPES } from "./universe";

/**
 * Graph — what this corpus knows.
 *
 * This is NOT the lineage canvas on /recall, and the distinction is the reason
 * both exist. Lineage answers "how do these recalled memories cause each
 * other" and only exists once you have run a query. This answers "what does
 * this corpus know": the typed entities and the typed relations between them,
 * explorable before any question has been asked. An analyst arrives here to
 * find out what is in there; they go to Recall when they already know what to
 * ask.
 *
 * The ontology is the content, so it is visible rather than implied — node
 * colour is entity type, straight from the `--node-*` tokens the design system
 * defines for exactly these four classes, and the legend names them.
 */

function Breadcrumb({
  clusterLabel,
  onRoot,
}: {
  clusterLabel: string | null;
  onRoot: () => void;
}) {
  return (
    <nav
      aria-label="Graph level"
      className="text-muted-foreground absolute top-3 left-4 z-10 flex items-center gap-1 text-[12px]"
    >
      {clusterLabel === null ? (
        <span>What this corpus knows</span>
      ) : (
        <>
          <button
            type="button"
            onClick={onRoot}
            className="hover:text-foreground focus-visible:ring-ring rounded transition-colors focus-visible:ring-2 focus-visible:outline-none"
          >
            What this corpus knows
          </button>
          <ChevronRight aria-hidden="true" className="size-3 shrink-0" />
          <span className="text-foreground">{clusterLabel}</span>
        </>
      )}
    </nav>
  );
}

export function GraphView({ reach }: { reach: Reachability }) {
  const { model, error, isFetching, profile } = useUniverse(reach);
  const selectEntity = useSession((s) => s.selectEntity);

  // Drill state lives here rather than in the canvas: the breadcrumb and the
  // canvas are two views of it, and the canvas is remounted per level.
  const [clusterId, setClusterId] = useState<number | null>(null);

  // A new corpus invalidates a drill path taken through the old one.
  useEffect(() => {
    setClusterId(null);
    selectEntity(null);
  }, [profile, selectEntity]);

  const baseLevel: Level | null = model ? levelFor(model) : null;
  const level: Level = clusterId !== null ? "entities" : (baseLevel ?? "entities");

  /** Entity types actually present, ordered with the four the design system
   *  names first — those carry their own hue; everything else shares one. */
  const legend = useMemo(() => {
    if (!model) return [];
    const counts = new Map<string, number>();
    for (const n of model.nodes) counts.set(n.type, (counts.get(n.type) ?? 0) + 1);
    const named = NAMED_ENTITY_TYPES.filter((t) => counts.has(t)).map((t) => ({
      label: t as string,
      token: entityTypeToken(t),
      count: counts.get(t)!,
    }));
    const otherCount = [...counts.entries()]
      .filter(([t]) => !(NAMED_ENTITY_TYPES as readonly string[]).includes(t))
      .reduce((a, [, c]) => a + c, 0);
    return otherCount > 0
      ? [...named, { label: "Other", token: "--chart-5", count: otherCount }]
      : named;
  }, [model]);

  if (reach.state !== "online") {
    return (
      <EmptyState
        size="page"
        title="Not connected"
        body="The knowledge graph is built from the memory server's entity store, which needs the server running."
      />
    );
  }

  if (profile === null) {
    return (
      <EmptyState
        size="page"
        title="No profile to show"
        body="The graph is per-profile — each one has its own entity store. This instance holds none yet."
      />
    );
  }

  if (error) {
    return (
      <EmptyState
        size="page"
        title="Graph failed to load"
        body="The entity universe did not come back. The graph endpoint returns the whole corpus in one response, so this fails all at once rather than partially."
      />
    );
  }

  if (!model) {
    return (
      <EmptyState
        size="page"
        title={isFetching ? "Building the graph" : "Nothing to draw"}
        body={
          isFetching
            ? "Fetching every entity and relation in this profile, then finding the communities in them. This is one request for the whole corpus."
            : "No entity universe came back for this profile."
        }
      />
    );
  }

  if (model.nodes.length === 0) {
    return (
      <EmptyState
        size="page"
        title="No entities yet"
        body="Nothing has been extracted into the knowledge graph for this profile. Entities appear as memories are written and the extraction pipeline types what is in them."
      />
    );
  }

  const cluster = clusterId !== null ? (model.clusters[clusterId] ?? null) : null;
  const shown =
    level === "clusters"
      ? `${model.clusters.length} cluster${model.clusters.length === 1 ? "" : "s"}`
      : cluster
        ? `${cluster.size} entit${cluster.size === 1 ? "y" : "ies"}`
        : `${model.nodes.length} entit${model.nodes.length === 1 ? "y" : "ies"}`;

  return (
    <section className="relative h-full min-h-0 min-w-0 overflow-hidden">
      <div aria-hidden="true" className="graticule pointer-events-none absolute inset-0" />

      <EntityCanvas
        // Remount on level change: the node set changes completely, and
        // reusing one simulation across that is how nodes end up carrying
        // positions from a graph they are no longer part of.
        key={`${level}:${clusterId ?? "root"}`}
        model={model}
        level={level}
        clusterId={clusterId}
        onDrillIn={(id) => {
          setClusterId(id);
          selectEntity(null);
        }}
      />

      <Breadcrumb
        clusterLabel={cluster ? cluster.label : null}
        onRoot={() => {
          setClusterId(null);
          selectEntity(null);
        }}
      />

      <div className="pointer-events-none absolute inset-x-4 bottom-3 z-10 flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
          {legend.map((l) => (
            <span key={l.label} className="text-muted-foreground flex items-center gap-1.5 text-[11px]">
              <span className="size-2 rounded-full" style={{ background: `var(${l.token})` }} />
              {l.label}
              <span className="text-muted-foreground/60">{l.count}</span>
            </span>
          ))}
        </div>
        <span className="text-muted-foreground/70 text-[11px]">
          {/* The corpus totals and the edge budget are stated, not hidden. The
              endpoint is uncapped, so if a cut happened it happened here, and a
              view that quietly drops two thirds of a graph is worse than one
              that admits it. */}
          {shown} of {model.totalEntities} · {model.totalConnections} relations
          {model.edgesDropped > 0 ? ` (weakest ${model.edgesDropped} not drawn)` : ""} ·{" "}
          {level === "clusters"
            ? "click a cluster to drill in"
            : "click an entity to inspect"}{" "}
          · scroll to zoom
        </span>
      </div>
    </section>
  );
}
