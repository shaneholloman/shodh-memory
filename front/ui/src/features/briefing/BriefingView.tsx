import { useMemo } from "react";

import { useCorpus } from "@/lib/api/corpus";
import type { Reachability } from "@/lib/api/health";

import { DotMap } from "./DotMap";
import type { PlacedPoint } from "./dot-map";
import {
  INDIA,
  INDIA_BOUNDS,
  INDIA_CELL,
  WORLD,
  WORLD_BOUNDS,
  WORLD_CELL,
} from "./geo-shapes";

/**
 * The briefing.
 *
 * Every other destination answers a question you already had. This one is the
 * only surface that tells you something without being asked — what changed
 * since you last looked, and what is worth looking at now. That is why it is
 * the landing view.
 *
 * It adds no fetches. Everything here is derived from the corpus cache entry
 * the geo and recall views already populate, so opening the briefing first
 * makes those two instant rather than making this one slow.
 */

/** Coordinates within ~55km collapse to one mark; below that the discs merge anyway. */
const CLUSTER_DEGREES = 0.5;

/** India's bounding box, used only to split the two maps' point sets. */
const INDIA_BOX = { minLon: 68, maxLon: 97.5, minLat: 6, maxLat: 37.5 };

type Located = { lat: number; lon: number };

/** Collapses nearby coordinates into weighted marks. */
function cluster(points: Located[]): PlacedPoint[] {
  const bins = new Map<string, { lon: number; lat: number; n: number }>();
  for (const { lat, lon } of points) {
    const key = `${Math.round(lat / CLUSTER_DEGREES)}:${Math.round(lon / CLUSTER_DEGREES)}`;
    const hit = bins.get(key);
    if (hit) hit.n += 1;
    else bins.set(key, { lon, lat, n: 1 });
  }
  // Sorted so the render order — and therefore which disc overlaps which — does
  // not depend on Map insertion order. Same discipline as the recall sorts.
  return [...bins.values()]
    .sort((a, b) => b.n - a.n || a.lon - b.lon || a.lat - b.lat)
    .map(({ lon, lat, n }) => [lon, lat, n] as PlacedPoint);
}

function inIndia({ lat, lon }: Located): boolean {
  return (
    lon >= INDIA_BOX.minLon &&
    lon <= INDIA_BOX.maxLon &&
    lat >= INDIA_BOX.minLat &&
    lat <= INDIA_BOX.maxLat
  );
}

export function BriefingView({ reach }: { reach: Reachability }) {
  const corpus = useCorpus(reach);
  const memories = corpus.data?.memories;

  const derived = useMemo(() => {
    const all = memories ?? [];
    const located: Located[] = [];
    for (const m of all) {
      // geo_location is [lat, lon, alt]; the renderer wants [lon, lat].
      if (m.geo_location) located.push({ lat: m.geo_location[0], lon: m.geo_location[1] });
    }

    const types = new Map<string, number>();
    for (const m of all) types.set(m.memory_type, (types.get(m.memory_type) ?? 0) + 1);

    return {
      total: all.length,
      located: located.length,
      world: cluster(located),
      india: cluster(located.filter(inIndia)),
      indiaCount: located.filter(inIndia).length,
      types: [...types.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])),
    };
  }, [memories]);

  // `isFetching` is true on background refetches too, so gate on "no data yet"
  // as well: a refresh must not blank a briefing the reader is already reading.
  if (corpus.isFetching && !corpus.data) {
    return (
      <section className="grid h-full place-items-center p-8 text-center">
        <div>
          <h2 className="text-base font-semibold">Reading the corpus</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            The briefing is derived, not stored — it is built from what is already here.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="h-full overflow-y-auto px-8 py-7">
      <header className="border-b border-border pb-5">
        <h1 className="max-w-2xl text-2xl leading-snug font-semibold">
          <span className="text-primary">{derived.total.toLocaleString()}</span> memories,{" "}
          <span className="text-primary">{derived.located.toLocaleString()}</span> of them
          carrying a place.
        </h1>

        {/* Memory kinds, not entity types. Labelled, because the two are one
            word apart and the mockup's headline figure was the other one. */}
        <ul className="mt-4 flex flex-wrap gap-x-6 gap-y-2 font-mono text-xs text-muted-foreground">
          {derived.types.slice(0, 6).map(([type, n]) => (
            <li key={type}>
              <span className="text-foreground">{n.toLocaleString()}</span> {type.toLowerCase()}
            </li>
          ))}
        </ul>
      </header>

      {/* The maps derive their height from their own width and the bounds'
          aspect, so an unbounded column gives India a ~900px plate. Cap the
          plate rather than the column: the captions and headings still want
          the full column width, and a briefing is read at arm's length. */}
      <div className="mt-7 grid gap-8 lg:grid-cols-2">
        <section>
          <h2 className="font-mono text-xs tracking-widest text-muted-foreground uppercase">
            The world
          </h2>
          <div className="mt-3 max-w-[26rem]">
            <DotMap
              shapes={WORLD}
              bounds={WORLD_BOUNDS}
              points={derived.world}
              cell={WORLD_CELL}
              label={`World map, ${derived.located} located memories`}
            />
          </div>
          <p className="mt-2 font-mono text-xs text-muted-foreground">
            {derived.located} of {derived.total} memories carry a place
          </p>
        </section>

        <section>
          <h2 className="font-mono text-xs tracking-widest text-muted-foreground uppercase">
            India
          </h2>
          <div className="mt-3 max-w-[20rem]">
            <DotMap
              shapes={INDIA}
              bounds={INDIA_BOUNDS}
              points={derived.india}
              cell={INDIA_CELL}
              label={`Map of India, ${derived.indiaCount} located memories`}
            />
          </div>
          <p className="mt-2 font-mono text-xs text-muted-foreground">
            {derived.indiaCount} memories · official boundary (LGD)
          </p>
        </section>
      </div>

      {/*
        DELIBERATELY ABSENT, and this is a note against re-adding it carelessly.

        The mockup's best element is the line that puts the corpus's worst
        property on the reader's first screen: "831 of 1,008 entities are typed
        Technology". That is GLiNER's ENTITY typing collapsing into one class,
        and it is invisible in the graph view because the graph draws whatever
        it is given.

        A first pass here rendered the same sentence from `memory_type`. That
        field is the memory KIND — Observation / Decision / Learning / Task, the
        four in the geo legend — so on the GDELT corpus it read "337 of 337
        typed Observation, 100% in one class, that is a typing failure". It is
        not. It is what a bulk import is, by construction. The instrument was
        reporting a defect that was not there, which is worse than showing
        nothing.

        Restoring it needs entity-type counts from the graph, not the corpus
        list — and the threshold has to be justified against a corpus where the
        typer is known to be working, or it just relabels normal skew as
        failure.
      */}

      <p className="mt-6 font-mono text-xs text-muted-foreground">
        Nothing here is generated — every figure is retrieved and traceable · Local · no network
        · Boundary: LGD via bharatlas · CC0-1.0 / CC-BY-4.0
      </p>
    </section>
  );
}
