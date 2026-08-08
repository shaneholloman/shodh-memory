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

/** One flagged memory, with the evidence that justifies flagging it. */
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
   * and understate it. What each row states — the distance, the two quantities,
   * the term count — is the honest presentation of its own magnitude, and
   * position in the list is the comparison.
   */
  deviation: number;
  /**
   * THE ROW'S OWN MAGNITUDE, AND NOTHING THAT IS TRUE OF EVERY ROW.
   *
   * This used to be one sentence per finding, and on a real corpus four
   * consecutive rows read "260 km / 259 km / 31 km / 18 km from the middle of
   * the cluster the other 35 placed memories form, which normally sit 2.4 km
   * out" — twelve identical words, three times over, for four numbers. The
   * shared half of every comparison belongs to the SECTION, which states it
   * once in `facts`; the row states only what distinguishes it.
   *
   * `value` is the measured quantity and is set in the mono face. `against` is
   * the short phrase naming what it is a quantity OF — never a restatement of
   * the section's baseline, and never omitted, because a bare "260 km" on a row
   * is the cryptic token this split exists to avoid.
   */
  value: string;
  against: string;
}

/**
 * What a lens says about itself, in the two registers the screen distinguishes.
 *
 * `facts` are short tokens rendered permanently under the section title — the
 * numbers that define where this lens drew its line on THIS corpus. They are
 * the section's half of every finding's comparison, stated once.
 *
 * `caveat` is a limitation that changes how the visible numbers should be read
 * — a spread that fell back to a weaker estimator, memories excluded from the
 * judgement entirely. It is ALWAYS rendered, never folded into an info panel:
 * the published finding on info tips is that most people never open them, and a
 * caveat nobody reads is worse than one that was never written, because the
 * screen then looks like it made a stronger claim than it did.
 *
 * `detail` is elaboration for the section's info panel — true, useful, and
 * never load-bearing. Nothing that qualifies a number on screen may go here.
 *
 * `insufficient` is a first-class outcome rather than an empty list, because
 * "no anomalies" and "not enough data to have an opinion" are different claims
 * and collapsing them into one blank panel would make the weaker one look like
 * the stronger one. The reason names the shortfall in numbers.
 */
interface LensBasis {
  facts: string[];
  caveat?: string;
  detail?: string;
}

export type LensResult =
  | ({ state: "findings"; findings: Finding[] } & LensBasis)
  | ({ state: "clear" } & LensBasis)
  | ({ state: "insufficient"; reason: string } & LensBasis);

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
      facts: [`${located.length} of ${memories.length} placed`, `${MIN_LOCATED} needed`],
      reason:
        located.length === 0
          ? "Nothing here carries coordinates, so there is no cluster to be outside of."
          : "With fewer, every point is as much the cluster as any other.",
      detail:
        "A memory only carries a position when whatever wrote it supplied one. Imported corpora usually do; session captures usually do not.",
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
      facts: [`${located.length} placed`, "no spread"],
      caveat:
        "Every placed memory sits the same distance from the centre, so there is nothing for one to stand outside of.",
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
      // Only the distance. "…from the middle of the cluster the other 35
      // placed memories form, which normally sit 2.4 km out" is true of every
      // row in this section and is stated once, above, in `facts`.
      value: formatKm(km),
      against: "from cluster centre",
    }))
    .sort((a, b) => b.deviation - a.deviation);

  // The spread token has to describe the estimator that actually produced the
  // number. Saying "half within" of a MEAN deviation would be a false statement
  // about the arithmetic, on a screen whose entire claim is that the arithmetic
  // is stated — so the non-robust case does not get the token at all. It gets a
  // caveat, on screen, in the state's own words.
  const facts = [
    `${located.length} placed`,
    `typical ${formatKm(typical)} out`,
    spread.robust ? `half within ${formatKm(spread.scale)}` : `spread ${formatKm(spread.scale)}`,
    `flagged above ${formatKm(cutoff)}`,
  ];

  const caveat = spread.robust
    ? undefined
    : `More than half of these share one exact position, so the median spread came out at zero and the average distance stands in for it — a looser line than the other measures draw.`;

  const detail = `Centre is the median latitude and longitude of the ${located.length} placed memories, and distance is great-circle from there. The cutoff is ${MODIFIED_Z_CUTOFF} robust scales past the typical distance — Iglewicz & Hoaglin's published modified-z cutoff, not a number tuned to this corpus. The remaining ${others} placed memories are the baseline every row is measured against.`;

  return findings.length > 0
    ? { state: "findings", findings, facts, caveat, detail }
    : { state: "clear", facts, caveat, detail };
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
        // Both numbers and the factor between them, which is the whole finding.
        // What is NOT said is which one is wrong: the arithmetic cannot know,
        // and a row that picked a side would be asserting past its evidence.
        value: `${formatValue(low.value)} vs ${formatValue(high.value)} ${unit}`,
        against: `${factor.toFixed(1)}× apart in one memory`,
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
        // The baseline stays on the row here, unlike the location lens: this
        // section mixes units, so "typical 38 t" is true of the tonnes rows and
        // false of the hours rows and cannot be hoisted into one section line.
        value: `${formatValue(entry.value)} ${unit}`,
        against: `typical ${formatValue(centre)} ${unit} across ${entries.length} memories`,
      });
    }
  }

  const parsedCount = perMemory.filter((p) => p.quantities.length > 0).length;

  const unitMethod =
    "Only a number written next to a known unit is read, and values are only ever compared within one unit. Percentages are left out on purpose: a percentage of one thing tells you nothing about a percentage of another.";

  if (parsedCount === 0) {
    return {
      state: "insufficient",
      facts: [`0 of ${memories.length} state a unit`],
      reason: "Nothing here writes a number next to a unit this measure recognises.",
      detail: `${unitMethod} Bare counts — containers, transactions, people — are not compared either, because two counts of different things share nothing but a number.`,
    };
  }

  const shortSummary = shortUnits
    .sort((a, b) => b.count - a.count)
    .slice(0, 4)
    .map((u) => `${u.unit} ${u.count}`)
    .join(", ");

  const facts = [
    `${parsedCount} of ${memories.length} state a unit`,
    comparedUnits.length > 0
      ? `range compared: ${comparedUnits.join(", ")}`
      : `no unit reaches ${MIN_UNIT_SAMPLES} memories`,
  ];

  // A scope limit, not a footnote: with no unit populous enough to have a
  // normal range, this section CANNOT have found a corpus-wide outlier, and a
  // reader who does not know that would read an empty section as "checked, and
  // nothing was out of range".
  const caveat =
    comparedUnits.length > 0
      ? undefined
      : "No unit yet appears in enough separate memories to establish a normal range, so only disagreements inside a single memory can be found.";

  const detail = `${unitMethod} A unit needs ${MIN_UNIT_SAMPLES} separate memories before a normal range is claimed for it${shortSummary ? `; the most common so far are ${shortSummary}` : ""}.`;

  findings.sort((a, b) => b.deviation - a.deviation);

  return findings.length > 0
    ? { state: "findings", findings, facts, caveat, detail }
    : { state: "clear", facts, caveat, detail };
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
      facts: [`${memories.length} stored`, `${MIN_CORPUS_FOR_ISOLATION} needed`],
      reason: "Below this, the answer says more about the corpus than about any memory in it.",
      detail:
        '"Connected to nothing else" needs something else to be connected to. Two memories count as connected when the extraction pulled the same term out of both.',
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
      value: `${own.size} ${own.size === 1 ? "term" : "terms"}`,
      // The terms themselves, not a count of them: this is the one lens whose
      // evidence is words rather than a quantity, and "12 terms, none shared"
      // with nothing to check is an assertion. Four is what fits on a row.
      against: `none shared — ${[...own].slice(0, 4).join(", ")}${own.size > 4 ? ", …" : ""}`,
    });
  }

  findings.sort((a, b) => b.deviation - a.deviation);

  const judged = memories.length - unextracted.length;
  const facts = [`${judged} of ${memories.length} have terms to compare`];

  // Which memories were judged at all is a limit on the finding, not a
  // footnote about it: a reader who thinks all 74 were checked reads an empty
  // section as a stronger result than it is.
  const caveat =
    unextracted.length > 0
      ? `${unextracted.length} of the ${memories.length} yielded no terms at all, so nothing can be said about what they connect to — they are left out rather than counted as isolated.`
      : undefined;

  const detail =
    "Two memories count as connected when the extraction pulled the same term out of both — including broad terms, so a memory reaches this list only by sharing nothing at all. The measure under-reports rather than over-reports, which is the safe direction on a screen like this one.";

  return findings.length > 0
    ? { state: "findings", findings, facts, caveat, detail }
    : { state: "clear", facts, caveat, detail };
}
