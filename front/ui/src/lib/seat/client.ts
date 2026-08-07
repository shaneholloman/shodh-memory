/**
 * The seat harness client. Every call goes to a same-origin `/seat/*` path:
 * the shodh-front Rust binary (and the vite dev proxy) strips the prefix and
 * injects the seat bearer token, so the browser holds no credential for the
 * seat, exactly as it holds no X-API-Key for the backend.
 *
 * Endpoints mirror seat/src/server.ts's route table; shapes are in types.ts.
 */

import { api, ApiError, NetworkError } from "@/lib/api/client";
import type {
  ConversationDetail,
  ConversationSummary,
  LedgerEntryView,
  ModelRef,
  ProviderInfo,
  SeatEvent,
  SeatHealthResponse,
  SeatModelInfo,
  SeatReachability,
} from "./types";

export async function probeSeat(signal?: AbortSignal): Promise<SeatReachability> {
  // /healthz answers 200 with the backend up and 503 with it down — the seat
  // itself is reachable in both cases, so a 503 is parsed, not treated as
  // "seat offline".
  try {
    const health = await api.get<SeatHealthResponse>("/seat/healthz", signal);
    return { state: "online", backendOk: health.backend.ok, backendDetail: health.backend.detail };
  } catch (err) {
    if (err instanceof ApiError) {
      if (err.status === 503) {
        try {
          const health = JSON.parse(err.body) as SeatHealthResponse;
          return {
            state: "online",
            backendOk: health.backend.ok,
            backendDetail: health.backend.detail,
          };
        } catch {
          return { state: "offline", detail: `seat answered 503: ${err.body.slice(0, 120)}` };
        }
      }
      // 502 is the front proxy saying it could not reach the seat process.
      return { state: "offline", detail: `seat answered ${err.status}` };
    }
    if (err instanceof NetworkError) return { state: "offline", detail: err.message };
    throw err;
  }
}

export function listModels(refresh: boolean, signal?: AbortSignal) {
  return api.get<{ models: SeatModelInfo[]; local_errors: Record<string, string> }>(
    `/seat/v1/models${refresh ? "?refresh=1" : ""}`,
    signal,
  );
}

export function listProviders(signal?: AbortSignal) {
  return api.get<{ providers: ProviderInfo[] }>("/seat/v1/providers", signal);
}

export async function setProviderKey(providerId: string, apiKey: string): Promise<ProviderInfo> {
  const res = await fetch(`/seat/v1/providers/${encodeURIComponent(providerId)}/key`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey }),
  });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
  return ((await res.json()) as { provider: ProviderInfo }).provider;
}

export async function clearProviderKey(providerId: string): Promise<ProviderInfo> {
  const res = await fetch(`/seat/v1/providers/${encodeURIComponent(providerId)}/key`, {
    method: "DELETE",
  });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
  return ((await res.json()) as { provider: ProviderInfo }).provider;
}

export function listConversations(userId: string, signal?: AbortSignal) {
  return api.get<{ conversations: ConversationSummary[] }>(
    `/seat/v1/conversations?user_id=${encodeURIComponent(userId)}`,
    signal,
  );
}

export function getConversation(id: string, signal?: AbortSignal) {
  return api.get<ConversationDetail>(`/seat/v1/conversations/${encodeURIComponent(id)}`, signal);
}

export function createConversation(body: {
  user_id: string;
  provider: string;
  model: string;
  system_prompt?: string;
}) {
  return api.post<ConversationSummary & { harness_user_id: string }>("/seat/v1/conversations", body);
}

export async function renameConversation(id: string, title: string): Promise<void> {
  const res = await fetch(`/seat/v1/conversations/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
}

export async function deleteConversation(id: string): Promise<void> {
  const res = await fetch(`/seat/v1/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
}

export async function changeModel(id: string, provider: string, model: string): Promise<ModelRef> {
  const res = await fetch(`/seat/v1/conversations/${encodeURIComponent(id)}/model`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, model }),
  });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
  return ((await res.json()) as { model: ModelRef }).model;
}

export function listLedger(conversationId: string | undefined, limit: number, signal?: AbortSignal) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (conversationId) params.set("conversation_id", conversationId);
  return api.get<{ events: LedgerEntryView[] }>(`/seat/v1/learning/events?${params}`, signal);
}

export function revertLedgerEvent(eventId: string) {
  return api.post<{ revert: { id: string } }>("/seat/v1/learning/revert", { event_id: eventId });
}

/**
 * POST a message and stream the SeatEvents back.
 *
 * `EventSource` cannot POST, so this parses the SSE frames off a fetch body
 * by hand: frames separated by a blank line, `event:`/`data:` fields,
 * comment lines (`: ping` heartbeats) and `retry:` ignored — the exact
 * grammar seat/src/server.ts emits. ~30 lines beats a dependency.
 *
 * Returns when the stream closes. Throws NetworkError if the connection
 * fails before or during the stream; events already delivered stand.
 */
export async function streamMessage(
  conversationId: string,
  text: string,
  onEvent: (event: SeatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`/seat/v1/conversations/${encodeURIComponent(conversationId)}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal,
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
    throw new NetworkError(cause);
  }
  if (!res.ok || !res.body) {
    throw new ApiError(res.status, await res.text().catch(() => ""));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (frame: string): void => {
    let data = "";
    for (const line of frame.split("\n")) {
      if (line.startsWith("data:")) data += line.slice(5).trimStart();
      // `event:` is redundant — the payload carries its own `type` — and
      // comments / `retry:` carry nothing.
    }
    if (!data) return;
    onEvent(JSON.parse(data) as SeatEvent);
  };

  for (;;) {
    let chunk: ReadableStreamReadResult<Uint8Array>;
    try {
      chunk = await reader.read();
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
      throw new NetworkError(cause);
    }
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    for (;;) {
      const boundary = buffer.indexOf("\n\n");
      if (boundary === -1) break;
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      dispatch(frame);
    }
  }
  if (buffer.trim()) dispatch(buffer);
}
