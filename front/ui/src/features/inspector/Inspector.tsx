import { useQueryClient } from "@tanstack/react-query";
import type { RecallResponse } from "@/lib/api";
import { useSession } from "@/stores/session";
import { ScoreBreakdown } from "./ScoreBreakdown";

/**
 * The Inspector — the single object-detail surface.
 *
 * WORKFLOWS.md: every list in the product feeds this one pane, and it always
 * offers the next hop. That is what makes a click lead somewhere instead of
 * terminating.
 *
 * It reads the recall result from the react-query cache under the same key the
 * result list used, rather than taking it as a prop or refetching. Recall is
 * expensive and its response already contains everything shown here —
 * attribution, lineage, facts. Asking again to render a detail view would pay
 * the full retrieval cost twice for data already in memory.
 *
 * Positioned absolutely rather than as a flex column so the d3 canvas can later
 * take the stage's full width with this floated over its right edge. Until then
 * `main` reserves the same width, so nothing is occluded.
 */

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mt-3">
      <div className="text-muted-foreground/70 mb-1 text-[10px] tracking-wide uppercase">
        {label}
      </div>
      {children}
    </div>
  );
}

function Empty({ body }: { body: string }) {
  return (
    <div className="px-4 py-3">
      <p className="text-muted-foreground text-[12px] leading-relaxed">{body}</p>
    </div>
  );
}

export function Inspector() {
  const profile = useSession((s) => s.profile);
  const query = useSession((s) => s.activeQuery);
  const selectedId = useSession((s) => s.selectedMemoryId);
  const select = useSession((s) => s.select);

  // Same key as ResultPane, read straight out of the cache.
  //
  // Deliberately `getQueryData` and not `useQuery({ enabled: false })`. That
  // form still *registers* a query for the key, and a registration carrying no
  // queryFn throws when nothing else has populated it — which is exactly the
  // first-paint case here, before any search has run. A cache read has no such
  // failure mode and cannot issue a request by accident.
  const data = useQueryClient().getQueryData<RecallResponse>(["recall", profile, query]);

  const memory = data?.memories.find((m) => m.id === selectedId);

  // Chain 2: the causal edges that connect this memory to others in the same
  // result set. The server returns these already — no extra endpoint.
  const lineage = (data?.lineage ?? []).filter(
    (e) => e.from === selectedId || e.to === selectedId,
  );

  return (
    <aside
      aria-label="Detail"
      className="border-border bg-card absolute top-12 right-0 bottom-0 z-20 w-[280px] overflow-y-auto border-l"
    >
      <header className="border-border border-b px-4 py-3">
        <h2 className="text-[12px] font-medium tracking-tight">Detail</h2>
        <p className="text-muted-foreground mt-0.5 text-[11px] leading-relaxed">
          Every belief traces back to the sessions it came from.
        </p>
      </header>

      {!memory ? (
        <Empty
          body={
            data
              ? "Select a result to see what it is, why it surfaced, and what it connects to."
              : "Search, then select a result to see why it surfaced."
          }
        />
      ) : (
        <>
          <div className="px-4 py-3">
            <p className="text-[13px] leading-relaxed">{memory.experience.content}</p>

            {memory.experience.tags.length > 0 ? (
              <Field label="Entities">
                <div className="flex flex-wrap gap-1">
                  {memory.experience.tags.map((t) => (
                    <span
                      key={t}
                      className="border-border text-muted-foreground rounded border px-1.5 py-0.5 text-[10px]"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </Field>
            ) : null}

            <Field label="Recorded">
              <p className="mono text-[11px]">
                {new Date(memory.created_at).toLocaleString()}
              </p>
            </Field>

            {memory.experience.geo_location ? (
              <Field label="Coordinates">
                <p className="mono text-[11px]">
                  {memory.experience.geo_location[0].toFixed(5)},{" "}
                  {memory.experience.geo_location[1].toFixed(5)}
                </p>
              </Field>
            ) : null}

            <Field label="Tier">
              <p className="mono text-[11px]">{memory.tier}</p>
            </Field>
          </div>

          {/* Chain 2 — the onward hop. Every edge here is a real causal
              relation the server returned, and selecting one moves the
              Inspector to the other end of it. */}
          {lineage.length > 0 ? (
            <section className="border-border border-t px-4 py-3">
              <h3 className="text-[12px] font-medium tracking-tight">What it connects to</h3>
              <div className="mt-2 flex flex-col gap-1">
                {lineage.map((e, i) => {
                  const otherId = e.from === selectedId ? e.to : e.from;
                  const other = data?.memories.find((m) => m.id === otherId);
                  const outgoing = e.from === selectedId;
                  return (
                    <button
                      key={`${e.from}-${e.to}-${i}`}
                      type="button"
                      onClick={() => select(otherId)}
                      disabled={!other}
                      className="hover:bg-accent/60 focus-visible:ring-ring rounded px-2 py-1.5 text-left transition-colors focus-visible:ring-2 focus-visible:outline-none disabled:opacity-50"
                    >
                      <span className="text-primary mono text-[10px]">
                        {outgoing ? "→" : "←"} {e.relation}
                      </span>
                      <span className="text-muted-foreground mt-0.5 line-clamp-2 block text-[11px] leading-relaxed">
                        {other?.experience.content ?? "Outside this result set"}
                      </span>
                    </button>
                  );
                })}
              </div>
            </section>
          ) : null}

          {memory.score_attribution ? (
            <ScoreBreakdown attr={memory.score_attribution} />
          ) : null}
        </>
      )}
    </aside>
  );
}
