import { HashRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Providers } from "./providers";
import { useReachability } from "./useReachability";
import { Sidebar, DESTINATIONS } from "@/components/layout/Sidebar";
import { TopBar, RAIL_OFFSET } from "@/components/layout/TopBar";
import { SearchField } from "@/components/layout/SearchField";
import { RecallView } from "@/features/recall/RecallView";
import { Inspector } from "@/features/inspector/Inspector";
import { Placeholder } from "@/components/layout/Placeholder";
import type { Reachability } from "@/lib/api";

/**
 * The shell.
 *
 * Nothing here is in normal flow except the content itself: the rail, the
 * header and the Inspector are all absolutely positioned, and `main` is offset
 * by their widths. That is what lets the rail expand over the stage instead of
 * shoving it, and what will let the graph canvas take the full width with
 * panels floated on top once it is ported. A flex row cannot do either.
 *
 * `HashRouter`, not `BrowserRouter`, and this is forced rather than chosen:
 * front/src/main.rs serves exactly three routes — `/`, the d3 asset, and the
 * `/api/{*path}` proxy — with no catch-all. A deep path like `/recall` would
 * 404 on refresh under history routing. The app also ships as one embedded
 * `index.html` inside the Rust binary, where a hash is the only routing that
 * survives being opened from anywhere.
 *
 * Two things from Gridline are deliberately absent. Its theme switch and the
 * MutationObserver behind it: this product is dark-only, so the control would
 * toggle nothing. Its version strip: there is no version feed behind it, and
 * chrome that displays invented data is worse than chrome that is absent.
 */

const INSPECTOR_OFFSET = "pr-[280px]";

/** The Inspector is the detail surface for recall results, so it accompanies
 *  that route only. On a destination with no selectable objects it could show
 *  nothing but an explanation of itself. */
const ROUTES_WITH_INSPECTOR = ["/recall"];

function Shell({ reach }: { reach: Reachability }) {
  const { pathname } = useLocation();
  const destination = DESTINATIONS.find((d) => d.path === pathname);
  const showInspector = ROUTES_WITH_INSPECTOR.includes(pathname) && reach.state === "online";

  return (
    <div className="relative h-svh min-h-0 overflow-hidden">
      <Sidebar reach={reach} />

      <TopBar title={destination?.label ?? "shodh"} reach={reach}>
        {pathname === "/recall" ? <SearchField reach={reach} /> : null}
      </TopBar>

      <main className={cn("h-full pt-12", RAIL_OFFSET, showInspector && INSPECTOR_OFFSET)}>
        <Routes>
          <Route path="/" element={<Navigate to="/recall" replace />} />
          <Route path="/recall" element={<RecallView reach={reach} />} />
          <Route path="/anomalies" element={<Placeholder id="anomalies" reach={reach} />} />
          <Route path="/tasks" element={<Placeholder id="tasks" reach={reach} />} />
          {/* A hash the app does not know is a typo or a stale link, not an
              error worth a page — send it home rather than showing a dead end. */}
          <Route path="*" element={<Navigate to="/recall" replace />} />
        </Routes>
      </main>

      {showInspector ? <Inspector /> : null}
    </div>
  );
}

/** Split from `App` because `useReachability` needs to be inside `Providers`
 *  to use the query client, and inside the router to be one instance. */
function Routed() {
  const reach = useReachability();
  return <Shell reach={reach} />;
}

export function App() {
  return (
    <Providers>
      <HashRouter>
        <Routed />
      </HashRouter>
    </Providers>
  );
}
