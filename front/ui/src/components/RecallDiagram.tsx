import { useState } from "react";
import { cn } from "@/lib/utils";

/**
 * How recall works, as a diagram that explains itself.
 *
 * Two problems this solves at once. The stage is dead space until the graph is
 * mounted — a raw error or a blank field is the first thing anyone sees. And
 * the vocabulary this product needs (spreading activation, causal chains,
 * provenance) cannot go in the chrome without turning it into jargon.
 *
 * Progressive disclosure handles both: the illustration reads at a glance, and
 * the explanation for any part appears only when someone asks for it by
 * hovering or focusing. Nothing is hidden behind a click that a person must
 * guess at — every part is a real focusable control, so this works from the
 * keyboard and reads correctly to a screen reader.
 */

type PartId = "cue" | "spread" | "surfaced" | "provenance";

const PARTS: {
  id: PartId;
  label: string;
  detail: string;
}[] = [
  {
    id: "cue",
    label: "Your cue",
    detail:
      "A few words are enough. Recall does not need the phrasing that was originally stored — it starts from meaning, not from matching text.",
  },
  {
    id: "spread",
    label: "Activation spreads",
    detail:
      "The cue lights up whatever it touches, and that activation travels along learned relations to neighbours. This is how a memory nobody searched for still surfaces.",
  },
  {
    id: "surfaced",
    label: "What surfaced",
    detail:
      "The memories that ended up most activated, ranked. Not just the closest match — the ones the rest of the graph agreed were relevant.",
  },
  {
    id: "provenance",
    label: "Where it came from",
    detail:
      "Every surfaced memory traces back to the session that recorded it, with a timestamp. Nothing here is generated; it is retrieved, so it can always be checked.",
  },
];

/**
 * One focusable region of the illustration.
 *
 * `hit` is not optional and not cosmetic. An SVG `<g>` has no area of its own —
 * pointer events land only on the strokes and fills inside it, and most of what
 * is drawn here is 1px lines and 4px circles. Without a transparent hit
 * rectangle the caption invites you to "point at any part" and then asks you to
 * hit a hairline. `fill="transparent"` is required rather than `fill="none"`,
 * which is not a paint and therefore takes no pointer events at all.
 */
function Hotspot({
  active,
  dimmed,
  label,
  hit,
  onEnter,
  onLeave,
  children,
}: {
  active: boolean;
  dimmed: boolean;
  label: string;
  hit: { x: number; y: number; w: number; h: number };
  onEnter: () => void;
  onLeave: () => void;
  children: React.ReactNode;
}) {
  return (
    <g
      role="button"
      tabIndex={0}
      aria-label={label}
      aria-pressed={active}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      onFocus={onEnter}
      onBlur={onLeave}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onEnter();
        }
        if (e.key === "Escape") onLeave();
      }}
      className={cn(
        "cursor-pointer outline-none transition-opacity duration-200",
        dimmed ? "opacity-30" : "opacity-100",
      )}
    >
      {/* Never drawn. Outlining it boxes the illustration into four rectangles
          and reads louder than the thing it is meant to reveal; the accent and
          the dimming of everything else already say which part is selected. */}
      <rect
        x={hit.x}
        y={hit.y}
        width={hit.w}
        height={hit.h}
        fill="transparent"
      />
      {children}
    </g>
  );
}

export function RecallDiagram() {
  const [active, setActive] = useState<PartId | null>(null);
  const shown = PARTS.find((p) => p.id === active);
  const dim = (id: PartId) => active !== null && active !== id;

  const accent = "var(--primary)";
  const line = "var(--border)";
  const nodeIdle = "#3a3d43";

  return (
    <div className="flex w-full max-w-2xl flex-col items-center gap-7 px-8">
      <svg
        viewBox="0 0 420 150"
        className="w-full"
        role="group"
        aria-label="How recall works. Four parts, each explains itself on hover or focus."
      >
        {/* ── the cue ─────────────────────────────────────────────── */}
        <Hotspot
          active={active === "cue"}
          dimmed={dim("cue")}
          label={PARTS[0].label}
          hit={{ x: 2, y: 48, w: 118, h: 46 }}
          onEnter={() => setActive("cue")}
          onLeave={() => setActive(null)}
        >
          <rect
            x="8"
            y="58"
            width="72"
            height="26"
            rx="6"
            fill="var(--card)"
            stroke={active === "cue" ? accent : line}
            strokeWidth="1"
          />
          <line x1="20" y1="71" x2="52" y2="71" stroke={nodeIdle} strokeWidth="3" strokeLinecap="round" />
          <line x1="80" y1="71" x2="118" y2="71" stroke={active === "cue" ? accent : line} strokeWidth="1.5" />
        </Hotspot>

        {/* ── spreading activation ────────────────────────────────── */}
        <Hotspot
          active={active === "spread"}
          dimmed={dim("spread")}
          label={PARTS[1].label}
          hit={{ x: 116, y: 26, w: 104, h: 92 }}
          onEnter={() => setActive("spread")}
          onLeave={() => setActive(null)}
        >
          {[
            ["124,71", "168,40"],
            ["124,71", "170,71"],
            ["124,71", "168,104"],
            ["170,71", "212,40"],
            ["170,71", "214,104"],
            ["168,40", "212,40"],
          ].map(([a, b], i) => {
            const [x1, y1] = a.split(",");
            const [x2, y2] = b.split(",");
            return (
              <line
                key={i}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={active === "spread" ? accent : line}
                strokeWidth="1"
              />
            );
          })}
          {[
            [124, 71, 5],
            [168, 40, 4],
            [170, 71, 4.5],
            [168, 104, 3.5],
            [212, 40, 3.5],
            [214, 104, 3],
          ].map(([cx, cy, r], i) => (
            <circle
              key={i}
              cx={cx}
              cy={cy}
              r={r}
              fill={active === "spread" ? accent : nodeIdle}
              className="transition-[fill] duration-200"
            />
          ))}
        </Hotspot>

        {/* ── surfaced memories ───────────────────────────────────── */}
        <Hotspot
          active={active === "surfaced"}
          dimmed={dim("surfaced")}
          label={PARTS[2].label}
          hit={{ x: 222, y: 36, w: 152, h: 76 }}
          onEnter={() => setActive("surfaced")}
          onLeave={() => setActive(null)}
        >
          <line x1="224" y1="71" x2="266" y2="71" stroke={active === "surfaced" ? accent : line} strokeWidth="1.5" />
          {[44, 66, 88].map((y, i) => (
            <rect
              key={y}
              x="272"
              y={y}
              width="96"
              height="16"
              rx="4"
              fill="var(--card)"
              stroke={active === "surfaced" ? accent : line}
              strokeWidth="1"
              opacity={1 - i * 0.22}
            />
          ))}
        </Hotspot>

        {/* ── provenance ──────────────────────────────────────────── */}
        <Hotspot
          active={active === "provenance"}
          dimmed={dim("provenance")}
          label={PARTS[3].label}
          hit={{ x: 92, y: 118, w: 240, h: 28 }}
          onEnter={() => setActive("provenance")}
          onLeave={() => setActive(null)}
        >
          <path
            d="M320 112 L320 132 L150 132"
            fill="none"
            stroke={active === "provenance" ? accent : line}
            strokeWidth="1"
            strokeDasharray="3 3"
          />
          <rect
            x="96"
            y="124"
            width="52"
            height="16"
            rx="4"
            fill="var(--card)"
            stroke={active === "provenance" ? accent : line}
            strokeWidth="1"
          />
        </Hotspot>
      </svg>

      {/* The caption region is reserved at a fixed height so revealing an
          explanation never reflows the illustration above it. */}
      <div className="flex min-h-[64px] w-full max-w-md flex-col items-center justify-start text-center">
        {shown ? (
          <>
            <p className="text-[13px] font-medium tracking-tight">
              {shown.label}
            </p>
            <p className="text-muted-foreground mt-1 text-[12px] leading-relaxed">
              {shown.detail}
            </p>
          </>
        ) : (
          <p className="text-muted-foreground/70 text-[12px]">
            How recall works — point at any part to see what it does.
          </p>
        )}
      </div>
    </div>
  );
}
