import { api, ApiError, NetworkError } from "./client";

/**
 * Reachability, authorization, and the profile list — from one request.
 *
 * `GET /api/users` is verified in src/handlers/router.rs and sits behind auth
 * (src/auth.rs::auth_middleware exempts only /health and /webhook/*), so a 401
 * here proves the key is wrong. /health would prove nothing about the key,
 * which is the failure this has to distinguish.
 *
 * The handler is `Json<Vec<String>>` — src/handlers/users.rs, `list_users`
 * returns `state.list_users()` — so the body is a plain array of profile ids.
 * The previous version of this probe fetched exactly this and threw the body
 * away, then the UI hardcoded an identity string next to it.
 */
export type Reachability =
  | { state: "online"; profiles: string[] }
  | { state: "unauthorized"; status: number }
  | { state: "offline"; detail: string };

export async function probeBackend(signal?: AbortSignal): Promise<Reachability> {
  try {
    const profiles = await api.get<string[]>("/api/users", signal);
    return { state: "online", profiles };
  } catch (err) {
    if (err instanceof ApiError) {
      return err.isAuthFailure
        ? { state: "unauthorized", status: err.status }
        : { state: "offline", detail: `backend returned ${err.status}` };
    }
    if (err instanceof NetworkError) return { state: "offline", detail: err.message };
    throw err;
  }
}
