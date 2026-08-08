import type { CorpusMemory } from "@/lib/api/corpus";

/**
 * The measures behind /anomalies.
 *
 * Every function here is pure, deterministic and arithmetic. No model is
 * consulted, no threshold is hand-set to a value that happens to suit one
 * corpus, and nothing is inferred that the corpus does not already state. That
 * is the whole claim the destination makes on screen: the memory finds these
 * itself, and the reader can check the finding against the text of the memory
 * it names.
 *
 * All three read the ONE corpus cache entry (lib/api/corpus.ts) — the listing
 * every other destination already loads. This feature issues no request of its
 * own, so opening it costs nothing beyond the arithmetic below.
 *
 * WHY ROBUST STATISTICS, NOT A MEAN. An anomaly detector built on a mean and a
 * standard deviation is defeated by the anomaly: one point 260 km away drags
 * the centre toward itself and inflates the spread, so the very thing being
 * looked for raises the bar it has to clear. The median and the median absolute
 * deviation have a 50% breakdown point — they are unmoved until half the corpus
 * is anomalous — which is the correct assumption for a store whose normal case
 * is "mostly ordinary".
 */

// =============================================================================
// SHARED SHAPE
// =============================================================================

/** One flagged memory, with the sentence that justifies flagging it. */
export interface Finding {
  memoryId: string;
  /** The memory's own text (a 500-char preview — `content_truncated` on the
   *  corpus row says when more exists). Shown so the finding can be checked. */
  content: string;
  /**
   * How far past the flag line this memory sits, in the lens's own terms.
   * Unitless and comparable only WITHIN a lens — a geo modified z-score and a
   * quantity ratio are different quantities and are never compared.
   *
   * Used to ORDER the findings within a section, and for nothing else. It is
   * never drawn and never printed: the magnitudes it ranks are unbounded (one
   * memory in this corpus sits 126 robust scales out, another 14), so any bar
   * scaled to the largest would render every other genuine finding as a stub
   * and understate it. What each row states in words — the distance, the two
   * quantities, the term count — is the honest presentation of its own
   * magnitude, and position in the list is the comparison.
   */
  deviation: number;
  /** The measure, stated in words, with the values it compared. */
  measure: string;
}

/**
 * What a lens has to say.
 *
 * `insufficient` is a first-class outcome rather than an empty list, because
 * "no anomalies" and "not enough data to have an opinion" are different claims
 * and collapsing them into one blank panel would make the weaker one look like
 * the stronger one. The reason string names the shortfall in numbers.
 */
export type LensResult =
  | { state: "findings"; findings: Finding[]; basis: string }
  | { state: "clear"; basis: string }
  | { state: "insufficient"; reason: string };

// =============================================================================
// ROBUST STATISTICS
// =============================================================================

function median(sorted: number[]): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  const mid = n >> 1;
  return n % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function medianOf(values: number[]): number {
  return median([...values].sort((a, b) => a - b));
}

/**
 * The robust scale estimate, and its fallbacks.
 *
 * `1.4826 * MAD` is the consistency-corrected MAD: for normally distributed
 * data it converges on the standard deviation, so a cutoff expressed in these
 * units means what a reader expects "3.5 sigma" to mean.
 *
 * MAD reaching exactly zero is not a corner case to be floored away — it is the
 * ordinary shape of a corpus where more than half the memories carry the SAME
 * coordinate (one warehouse, one office). When it happens the MAD says nothing,
 * so the estimator falls back to the mean absolute deviation from the median,
 * which is smaller and less robust but still real. If that is zero too, every
 * value is identical and there is genuinely no spread to measure — `null`, and
 * the caller must say so rather than divide.
 */
function robustScale(values: number[], centre: number): { scale: number; robust: boolean } | null {
  const deviations = values.map((v) => Math.abs(v - centre));
  const mad = medianOf(deviations);
  if (mad > 0) return { scale: 1.4826 * mad, robust: true };
  const mean = deviations.reduce((a, b) => a + b, 0) / deviations.length;
  if (mean > 0) return { scale: mean, robust: false };
  return null;
}

/**
 * The flag line, in robust-scale units.
 *
 * 3.5 is Iglewicz & Hoaglin's published cutoff for the modified z-score
 * (NIST/SEMATECH e-Handbook §1.3.5.17), not a number chosen because it
 * separates this corpus nicely. It is stated here once and used by both lenses
 * that need a cutoff, so neither can be quietly tuned to a demo.
 */
const MODIFIED_Z_CUTOFF = 3.5;

// =============================================================================
// LENS 1 — OFF-PATTERN LOCATION
// =============================================================================

/** Below this there is no cluster to be far from, only a handful of points. */
export const MIN_LOCATED = 5;

const EARTH_RADIUS_KM = 6371.0088;

function toRadians(degrees: number): number {
  return (degrees * Math.PI) / 180;
}

/** Great-circle distance in kilometres. Haversine, which is numerically stable
 *  at the short distances that dominate a real corpus (points inside one city)
 *  where the spherical law of cosines loses precision. */
function greatCircleKm(aLat: number, aLon: number, bLat: number, bLon: number): number {
  const dLat = toRadians(bLat - aLat);
  const dLon = toRadians(bLon - aLon);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRadians(aLat)) * Math.cos(toRadians(bLat)) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.min(1, Math.sqrt(h)));
}

function formatKm(km: number): string {
  if (km < 1) return `${Math.round(km * 1000)} m`;
  if (km < 10) return `${km.toFixed(1)} km`;
  return `${Math.round(km)} km`;
}

/**
 * Memories whose coordinates sit outside the corpus's own spatial cluster.
 *
 * The reference is built from the corpus, never from a map or a place name:
 * the centre is the component-wise median of latitude and longitude — the
 * coordinate half the memories are north of and half are east of — and the
 * spread is the MAD of each memory's great-circle distance to that centre. A
 * memory is flagged when its distance is more than 3.5 robust scales past the
 * typical one.
 *
 * The centre is computed over ALL located memories including the outliers,
 * which is safe precisely because it is a median: a point 260 km out moves the
 * median by at most one rank position, whereas it would drag a mean most of the
 * way to itself.
 */
export function offPatternLocations(memories: CorpusMemory[]): LensResult {
  const located = memories.filter(
    (m): m is CorpusMemory & { geo_location: [number, number, number] } =>
      Array.isArray(m.geo_location) &&
      Number.isFinite(m.geo_location[0]) &&
      Number.isFinite(m.geo_location[1]),
  );

  if (located.length < MIN_LOCATED) {
    return {
      state: "insufficient",
      reason:
        located.length === 0
          ? `Nothing here carries coordinates. A memory only has a position when whatever wrote it supplied one, and none of these ${memories.length} did — so there is no cluster to be outside of.`
          : `A baseline needs at least ${MIN_LOCATED} placed memories to be worth trusting; this profile has ${located.length}. With fewer, every point is as much "the cluster" as any other.`,
    };
  }

  const centreLat = medianOf(located.map((m) => m.geo_location[0]));
  const centreLon = medianOf(located.map((m) => m.geo_location[1]));
  const distances = located.map((m) =>
    greatCircleKm(centreLat, centreLon, m.geo_location[0], m.geo_location[1]),
  );

  const typical = medianOf(distances);
  const spread = robustScale(distances, typical);
  const others = located.length - 1;

  if (spread === null) {
    return {
      state: "clear",
      basis: `All ${located.length} placed memories sit the same distance from the middle of the group, so there is no spread for anything to stand outside of.`,
    };
  }

  const cutoff = typical + MODIFIED_Z_CUTOFF * spread.scale;

  const findings: Finding[] = located
    .map((m, i) => ({ memory: m, km: distances[i] }))
    .filter(({ km }) => km > cutoff)
    .map(({ memory, km }) => ({
      memoryId: memory.id,
      content: memory.content,
      deviation: (km - typical) / spread.scale,
      measure: `${formatKm(km)} from the middle of the cluster the other ${others} placed memories form, which normally sit ${formatKm(typical)} out.`,
    }))
    .sort((a, b) => b.deviation - a.deviation);

  // The spread clause has to describe the estimator that actually produced the
  // number. Saying "half of them fall within" of a MEAN deviation would be a
  // false statement about the arithmetic, on a screen whose entire claim is
  // that the arithmetic is stated.
  const spreadClause = spread.robust
    ? `and half of them fall within ${formatKm(spread.scale)} of that`
    : `and more than half of them share one exact position — so the usual half-of-them spread came out at zero and the average distance from the centre, ${formatKm(spread.scale)}, stands in for it, which is a looser reading`;

  const basis =
    `Centre and spread come from the ${located.length} placed memories themselves: the middle one sits ${formatKm(typical)} from the group's centre, ${spreadClause}. ` +
    `Anything more than ${formatKm(cutoff)} out is listed.`;

  return findings.length > 0
    ? { state: "findings", findings, basis }
    : {
        state: "clear",
        basis: `${basis} Nothing is.`,
      };
}

// =============================================================================
// LENS 2 — QUANTITY OUTLIER
// =============================================================================

/**
 * The units this reads, and the exact spelling variants that map to each.
 *
 * A whitelist, not a pattern, because "number followed by a word" matches
 * "4,102 truck", "2,900 import" and "12 containers" — counts of things, whose
 * magnitudes have nothing to do with each other. Only a closed set of units
 * whose values are genuinely comparable earns a comparison.
 *
 * Values are NEVER converted between units, not even within one dimension.
 * "300 metres" (a vessel's length) and "12.8 metres" (a draft) are already
 * different quantities that happen to share a unit; folding kilometres in
 * beside them would compound the problem rather than add information. Same
 * unit, same spelling family, or no comparison.
 *
 * PERCENTAGES ARE DELIBERATELY ABSENT. `%` is the most common "unit" in a real
 * corpus and the least comparable one: 91% chassis availability, 3.1% exam rate
 * and 14% growth share a symbol and nothing else. Pooling them would
 * manufacture an outlier out of a bare coincidence of notation.
 */
const UNIT_SPELLINGS: Record<string, string> = {
  kg: "kg",
  kilogram: "kg",
  kilograms: "kg",
  t: "tonnes",
  ton: "tonnes",
  tons: "tonnes",
  tonne: "tonnes",
  tonnes: "tonnes",
  lb: "lb",
  lbs: "lb",
  m: "metres",
  meter: "metres",
  meters: "metres",
  metre: "metres",
  metres: "metres",
  km: "km",
  kilometer: "km",
  kilometers: "km",
  kilometre: "km",
  kilometres: "km",
  mi: "miles",
  mile: "miles",
  miles: "miles",
  ms: "ms",
  millisecond: "ms",
  milliseconds: "ms",
  sec: "seconds",
  secs: "seconds",
  second: "seconds",
  seconds: "seconds",
  min: "minutes",
  mins: "minutes",
  minute: "minutes",
  minutes: "minutes",
  h: "hours",
  hr: "hours",
  hrs: "hours",
  hour: "hours",
  hours: "hours",
  day: "days",
  days: "days",
  week: "weeks",
  weeks: "weeks",
  teu: "TEU",
  teus: "TEU",
  knot: "knots",
  knots: "knots",
};

/** A number, then optionally a hyphen (`4,000-hour`), then a word. The word is
 *  only kept if it is a known unit, and the number is only kept if nothing
 *  alphanumeric or hyphenated runs into it from the left — which is what stops
 *  `MSKU-4471820` and `T-118` from being read as quantities. */
const QUANTITY_PATTERN = /(\d[\d,]*(?:\.\d+)?)\s*-?\s*([A-Za-z]+)/g;

interface Quantity {
  value: number;
  unit: string;
  /** The substring as it was written, so the finding can quote the memory. */
  literal: string;
}

/** Exported for the same reason the lenses are: this is the only place a
 *  number becomes a claim, and it must be testable in isolation. */
export function parseQuantities(text: string): Quantity[] {
  const found: Quantity[] = [];
  QUANTITY_PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = QUANTITY_PATTERN.exec(text)) !== null) {
    const preceding = match.index > 0 ? text[match.index - 1] : "";
    if (/[A-Za-z0-9-]/.test(preceding)) continue;
    const unit = UNIT_SPELLINGS[match[2].toLowerCase()];
    if (unit === undefined) continue;
    const value = Number.parseFloat(match[1].replace(/,/g, ""));
    if (!Number.isFinite(value) || value <= 0) continue;
    found.push({ value, unit, literal: `${match[1]} ${match[2]}` });
  }
  return found;
}

/** A unit needs this many memories carrying it before the corpus can claim to
 *  know what a normal value looks like. Four samples cannot establish a spread
 *  that a fifth could be an outlier against. */
export const MIN_UNIT_SAMPLES = 5;

function formatValue(value: number): string {
  return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

/**
 * Numbers that do not fit — measured two ways, both unit-matched.
 *
 * WITHIN ONE MEMORY: two values in the same unit written in the same memory,
 * reported with the factor between them. This is a complete measure on its own
 * and needs no threshold, which is exactly why it is here: a declared weight
 * and a weighed weight in one sentence are a disagreement whatever the corpus
 * around them looks like. The finding states both numbers and the factor and
 * leaves the judgement to the reader — it does not assert which one is wrong,
 * because the arithmetic cannot know.
 *
 * ACROSS THE CORPUS: a value more than 3.5 robust scales from the median of
 * every other value written in that same unit. This needs a population, so it
 * only runs on units carried by at least `MIN_UNIT_SAMPLES` memories, and the
 * result says which units qualified.
 *
 * The two are complementary rather than redundant: the first catches a
 * contradiction inside one statement, the second catches a statement that is
 * out of step with every other one.
 */
export function quantityOutliers(memories: CorpusMemory[]): LensResult {
  const perMemory = memories.map((m) => ({ memory: m, quantities: parseQuantities(m.content) }));

  const findings: Finding[] = [];

  // --- within one memory ---------------------------------------------------
  for (const { memory, quantities } of perMemory) {
    const byUnit = new Map<string, Quantity[]>();
    for (const q of quantities) {
      const bucket = byUnit.get(q.unit);
      if (bucket) bucket.push(q);
      else byUnit.set(q.unit, [q]);
    }
    for (const [unit, values] of byUnit) {
      if (values.length < 2) continue;
      const low = values.reduce((a, b) => (a.value <= b.value ? a : b));
      const high = values.reduce((a, b) => (a.value >= b.value ? a : b));
      if (high.value === low.value) continue;
      const factor = high.value / low.value;
      findings.push({
        memoryId: memory.id,
        content: memory.content,
        deviation: Math.log10(factor),
        measure: `Two ${unit} figures in one memory that do not agree: ${formatValue(low.value)} and ${formatValue(high.value)} — a factor of ${factor.toFixed(1)}.`,
      });
    }
  }

  // --- across the corpus ---------------------------------------------------
  const population = new Map<string, { value: number; memory: CorpusMemory }[]>();
  for (const { memory, quantities } of perMemory) {
    // One entry per unit per memory: a memory that writes the same unit twice
    // must not count twice toward the population it is being judged against.
    for (const unit of new Set(quantities.map((q) => q.unit))) {
      const highest = quantities
        .filter((q) => q.unit === unit)
        .reduce((a, b) => (a.value >= b.value ? a : b));
      const bucket = population.get(unit);
      if (bucket) bucket.push({ value: highest.value, memory });
      else population.set(unit, [{ value: highest.value, memory }]);
    }
  }

  const comparedUnits: string[] = [];
  const shortUnits: { unit: string; count: number }[] = [];
  for (const [unit, entries] of population) {
    if (entries.length < MIN_UNIT_SAMPLES) {
      shortUnits.push({ unit, count: entries.length });
      continue;
    }
    comparedUnits.push(unit);
    const values = entries.map((e) => e.value);
    const centre = medianOf(values);
    const spread = robustScale(values, centre);
    if (spread === null) continue;
    for (const entry of entries) {
      const z = Math.abs(entry.value - centre) / spread.scale;
      if (z <= MODIFIED_Z_CUTOFF) continue;
      findings.push({
        memoryId: entry.memory.id,
        content: entry.memory.content,
        deviation: z,
        measure: `${formatValue(entry.value)} ${unit}, against a typical ${formatValue(centre)} ${unit} across the ${entries.length} memories that state one.`,
      });
    }
  }

  const parsedCount = perMemory.filter((p) => p.quantities.length > 0).length;

  if (parsedCount === 0) {
    return {
      state: "insufficient",
      reason: `None of these ${memories.length} memories states a number with a unit attached. Counts on their own — containers, transactions, people — are not compared, because two counts of different things share nothing but a number.`,
    };
  }

  const shortSummary = shortUnits
    .sort((a, b) => b.count - a.count)
    .slice(0, 4)
    .map((u) => `${u.unit} (${u.count})`)
    .join(", ");

  const basis =
    `Read from the text of ${parsedCount} of the ${memories.length} memories: only a number written next to a known unit counts, values are only ever compared with the same unit, and percentages are left out because a percentage of one thing tells you nothing about a percentage of another. ` +
    (comparedUnits.length > 0
      ? `${comparedUnits.join(", ")} appear in enough memories (${MIN_UNIT_SAMPLES} or more) to have a normal range worth comparing against.`
      : `No unit yet appears in the ${MIN_UNIT_SAMPLES} separate memories needed to establish a normal range${shortSummary ? ` — the most common so far are ${shortSummary}` : ""}. Until one does, only disagreements inside a single memory can be found.`);

  findings.sort((a, b) => b.deviation - a.deviation);

  return findings.length > 0
    ? { state: "findings", findings, basis }
    : { state: "clear", basis: `${basis} Nothing stands out.` };
}

// =============================================================================
// LENS 3 — ISOLATED MEMORY
// =============================================================================

/** Fewer than this and "shares nothing with the others" is a statement about
 *  the corpus being tiny, not about the memory. */
export const MIN_CORPUS_FOR_ISOLATION = 5;

/**
 * Memories that share no extracted term with any other memory.
 *
 * WHERE THE TERMS COME FROM. The corpus row's `tags` are the memory's extracted
 * entities verbatim — `tags: m.experience.entities.clone()`
 * (src/handlers/crud.rs:361 and :486). They are what the extraction pipeline
 * pulled out of the text, so this lens reads the graph's own view of what a
 * memory is about rather than re-deriving one here.
 *
 * WHY NOT THE UNIVERSE ENDPOINT. `GET /api/graph/{user}/universe` is the truer
 * picture of connectivity — it carries typed relations and their strengths —
 * but a `UniverseStar` has no memory id on it at all (lib/api/graph.ts:41),
 * so an entity there cannot be walked back to the memories that mention it.
 * Recovering that hop means one `POST /api/graph/traverse` per entity, and it
 * would still only see entities that formed an edge, which needs two entities
 * in one memory (graph.ts:134-139) — precisely excluding the sparse memories
 * this lens exists to find. The corpus listing already carries the extraction
 * result for every memory, including the sparse ones, in the request the page
 * has already made.
 *
 * THIS MEASURE UNDER-REPORTS, AND THAT IS THE SAFE DIRECTION. Extraction emits
 * broad terms alongside specific ones, so two memories can share a term without
 * being about the same thing. Any such term still counts as a connection here,
 * which means a memory reaches this list only by sharing nothing at all — a
 * false alarm requires every one of its terms to be unique, while a missed
 * isolate only requires one loose term in common. Over-reporting on an
 * anomalies screen costs trust; under-reporting costs a finding.
 */
export function isolatedMemories(memories: CorpusMemory[]): LensResult {
  if (memories.length < MIN_CORPUS_FOR_ISOLATION) {
    return {
      state: "insufficient",
      reason: `"Connected to nothing else" needs something else to be connected to. This profile holds ${memories.length} ${memories.length === 1 ? "memory" : "memories"}; below ${MIN_CORPUS_FOR_ISOLATION} the answer says more about the corpus than about any memory in it.`,
    };
  }

  const terms = (m: CorpusMemory) =>
    new Set(m.tags.map((t) => t.trim().toLowerCase()).filter((t) => t.length > 0));

  // How many DISTINCT memories each term appears in — not how many times it was
  // extracted, so a term repeated within one memory never looks like a link.
  const reach = new Map<string, number>();
  for (const m of memories) {
    for (const term of terms(m)) reach.set(term, (reach.get(term) ?? 0) + 1);
  }

  const unextracted: CorpusMemory[] = [];
  const findings: Finding[] = [];

  for (const m of memories) {
    const own = terms(m);
    if (own.size === 0) {
      unextracted.push(m);
      continue;
    }
    const shared = [...own].filter((t) => (reach.get(t) ?? 0) > 1);
    if (shared.length > 0) continue;
    findings.push({
      memoryId: m.id,
      content: m.content,
      // More unique terms is stronger isolation: a memory naming twelve things,
      // none of which any other memory names, is further out than one naming
      // two. Ordering only — never shown as a number.
      deviation: own.size,
      measure: `All ${own.size} of the terms taken from this memory — ${[...own].slice(0, 4).join(", ")}${own.size > 4 ? ", …" : ""} — appear nowhere else in the ${memories.length} memories here.`,
    });
  }

  findings.sort((a, b) => b.deviation - a.deviation);

  const judged = memories.length - unextracted.length;
  const caveat =
    unextracted.length > 0
      ? ` ${unextracted.length} of the ${memories.length} yielded no terms at all, so nothing can be said about what they connect to — they are left out rather than counted as isolated.`
      : "";

  const basis = `Two memories count as connected when the extraction pulled the same term out of both. ${judged} of the ${memories.length} memories here have terms to compare; these share none of theirs with any other.${caveat}`;

  if (findings.length === 0) {
    return {
      state: "clear",
      basis: `Two memories count as connected when the extraction pulled the same term out of both. Every one of the ${judged} memories with terms shares at least one with another.${caveat}`,
    };
  }

  return { state: "findings", findings, basis };
}
