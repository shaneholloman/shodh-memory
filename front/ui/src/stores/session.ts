import { create } from "zustand";

/**
 * The two pieces of state that are genuinely global.
 *
 * WORKFLOWS.md: "Selection is global state, not per-panel. One selected object
 * at a time." Every list in the product feeds one Inspector, so the selected id
 * cannot live inside whichever list produced it — the graph, the result list
 * and the lineage edges all set and read the same value.
 *
 * The active profile is global for a harder reason. Every recall carries a
 * `user_id`, and `MultiUserMemoryManager::get_user_memory` (src/handlers/state.rs)
 * *creates* a store on demand for an id it has not seen — it does not 404. So a
 * wrong or invented profile does not fail loudly; it silently provisions an
 * empty RocksDB directory and returns no results. The active profile is
 * therefore only ever set from the list the server itself returned, and is
 * `null` until one arrives.
 */
interface SessionState {
  /** Always a value the server listed in `GET /api/users`. Never invented. */
  profile: string | null;
  /** Id of the memory currently open in the Inspector. */
  selectedMemoryId: string | null;
  /** The query the results on screen came from, for the Inspector's header. */
  activeQuery: string;

  setProfile: (profile: string | null) => void;
  select: (memoryId: string | null) => void;
  setActiveQuery: (query: string) => void;
  /** Adopt the server's list: keep the current profile if it is still offered,
   *  otherwise fall back to the first. Prevents a stale selection surviving a
   *  backend restart that dropped it. */
  reconcileProfiles: (profiles: string[]) => void;
}

export const useSession = create<SessionState>((set) => ({
  profile: null,
  selectedMemoryId: null,
  activeQuery: "",

  setProfile: (profile) => set({ profile, selectedMemoryId: null }),
  select: (selectedMemoryId) => set({ selectedMemoryId }),
  setActiveQuery: (activeQuery) => set({ activeQuery }),

  reconcileProfiles: (profiles) =>
    set((s) => {
      if (profiles.length === 0) return { profile: null, selectedMemoryId: null };
      if (s.profile && profiles.includes(s.profile)) return s;
      return { profile: profiles[0], selectedMemoryId: null };
    }),
}));
