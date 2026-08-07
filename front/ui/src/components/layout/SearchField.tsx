import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import type { Reachability } from "@/lib/api";
import { useSession } from "@/stores/session";

/**
 * The search control.
 *
 * Submit-driven. Recall runs vector, keyword and graph retrieval and then fuses
 * and re-ranks them; firing that on every keystroke would spend the work
 * repeatedly to answer prefixes nobody asked about.
 *
 * The draft is local and the committed query is global, because they are
 * genuinely different things: the draft is what someone is typing, the
 * committed query is what the results on screen are an answer to. Keeping them
 * as one value makes the result list contradict its own heading mid-word.
 */
export function SearchField({ reach }: { reach: Reachability }) {
  const profile = useSession((s) => s.profile);
  const activeQuery = useSession((s) => s.activeQuery);
  const setActiveQuery = useSession((s) => s.setActiveQuery);
  const [draft, setDraft] = useState(activeQuery);

  // Switching profile switches corpus, so a query committed against the old one
  // is no longer what is on screen. Clearing the draft with it avoids offering
  // to re-run a search whose results would silently be from elsewhere.
  useEffect(() => setDraft(""), [profile]);

  const usable = reach.state === "online" && profile !== null;

  return (
    <form
      role="search"
      onSubmit={(e) => {
        e.preventDefault();
        setActiveQuery(draft.trim());
      }}
      className="flex min-w-0 flex-1 justify-end"
    >
      <input
        type="search"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        disabled={!usable}
        placeholder={usable ? "Search memory…" : "Search unavailable"}
        aria-label="Search memory"
        className={cn(
          // Shrinks rather than pushing the title off. A fixed 280px crowded
          // the heading out of the bar below ~480px.
          "bg-background border-input placeholder:text-muted-foreground w-full max-w-[280px] min-w-0",
          "rounded-md border px-2.5 py-1.5 text-[13px]",
          "focus-visible:border-ring focus-visible:ring-ring/40 focus-visible:ring-2 focus-visible:outline-none",
          "disabled:cursor-not-allowed disabled:opacity-50",
        )}
      />
    </form>
  );
}
