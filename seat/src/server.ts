/**
 * Seat HTTP server (node:http, no framework).
 *
 * Endpoints:
 *   GET    /healthz
 *   GET    /v1/models[?refresh=1]
 *   POST   /v1/conversations                      { user_id, provider, model, system_prompt? }
 *   GET    /v1/conversations/{id}
 *   POST   /v1/conversations/{id}/messages        { text }  → SSE stream of SeatEvents
 *   PATCH  /v1/conversations/{id}/model           { provider, model }
 *   GET    /v1/learning/events[?limit&conversation_id]
 *   POST   /v1/learning/revert                    { event_id }
 *
 * Auth: optional bearer token (mandatory for non-loopback binds, enforced at
 * config load). Provider credentials never appear in any response or event.
 */

import * as http from "node:http";
import type { ShodhBackend } from "./backend.js";
import type { SeatConfig } from "./config.js";
import { Conversation, ConversationBusyError, type ConversationDeps, UnknownModelError } from "./conversation.js";
import type { SeatEvent } from "./events.js";
import { LedgerError, type LearningLedger } from "./ledger.js";
import type { McpHost } from "./mcp.js";
import type { ModelRegistry } from "./models-registry.js";

const MAX_BODY_BYTES = 1_048_576;
const SSE_HEARTBEAT_MS = 15_000;

interface CreateConversationBody {
	user_id?: string;
	provider?: string;
	model?: string;
	system_prompt?: string;
}

class HttpError extends Error {
	readonly status: number;

	constructor(status: number, message: string) {
		super(message);
		this.status = status;
	}
}

function readBody(request: http.IncomingMessage): Promise<string> {
	return new Promise((resolve, reject) => {
		let size = 0;
		const chunks: Buffer[] = [];
		request.on("data", (chunk: Buffer) => {
			size += chunk.length;
			if (size > MAX_BODY_BYTES) {
				reject(new HttpError(413, "Request body too large"));
				request.destroy();
				return;
			}
			chunks.push(chunk);
		});
		request.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
		request.on("error", reject);
	});
}

function parseJson<T>(raw: string): T {
	if (!raw.trim()) throw new HttpError(400, "Empty request body");
	try {
		return JSON.parse(raw) as T;
	} catch {
		throw new HttpError(400, "Invalid JSON body");
	}
}

function sendJson(response: http.ServerResponse, status: number, payload: unknown): void {
	const body = JSON.stringify(payload);
	response.writeHead(status, {
		"Content-Type": "application/json; charset=utf-8",
		"Content-Length": Buffer.byteLength(body),
	});
	response.end(body);
}

export interface SeatServerDeps {
	config: SeatConfig;
	backend: ShodhBackend;
	registry: ModelRegistry;
	ledger: LearningLedger;
	mcpHost: McpHost;
}

export class SeatServer {
	private readonly deps: SeatServerDeps;
	private readonly conversations = new Map<string, Conversation>();
	private readonly server: http.Server;

	constructor(deps: SeatServerDeps) {
		this.deps = deps;
		this.server = http.createServer((request, response) => {
			this.route(request, response).catch((error) => {
				const status = error instanceof HttpError ? error.status : 500;
				const message = error instanceof Error ? error.message : String(error);
				if (!response.headersSent) {
					sendJson(response, status, { error: message });
				} else {
					response.end();
				}
			});
		});
	}

	listen(): Promise<void> {
		return new Promise((resolve, reject) => {
			this.server.once("error", reject);
			this.server.listen(this.deps.config.port, this.deps.config.host, () => resolve());
		});
	}

	close(): Promise<void> {
		return new Promise((resolve) => {
			for (const conversation of this.conversations.values()) conversation.abort();
			this.server.close(() => resolve());
		});
	}

	private authorize(request: http.IncomingMessage): void {
		const token = this.deps.config.authToken;
		if (!token) return;
		const header = request.headers.authorization ?? "";
		if (header !== `Bearer ${token}`) throw new HttpError(401, "Unauthorized");
	}

	private conversation(id: string): Conversation {
		const conversation = this.conversations.get(id);
		if (!conversation) throw new HttpError(404, `Unknown conversation: ${id}`);
		return conversation;
	}

	private async route(request: http.IncomingMessage, response: http.ServerResponse): Promise<void> {
		const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "localhost"}`);
		const method = request.method ?? "GET";
		const segments = url.pathname.split("/").filter(Boolean);

		if (method === "GET" && url.pathname === "/healthz") {
			await this.handleHealth(response);
			return;
		}

		this.authorize(request);

		if (method === "GET" && url.pathname === "/v1/models") {
			await this.handleModels(url, response);
			return;
		}
		if (method === "POST" && url.pathname === "/v1/conversations") {
			await this.handleCreateConversation(request, response);
			return;
		}
		if (segments[0] === "v1" && segments[1] === "conversations" && segments[2]) {
			const conversation = this.conversation(segments[2]);
			if (method === "GET" && segments.length === 3) {
				sendJson(response, 200, {
					conversation_id: conversation.id,
					user_id: conversation.userId,
					harness_user_id: conversation.harnessUserId,
					model: conversation.model,
					created_at: conversation.createdAt.toISOString(),
					busy: conversation.isStreaming,
					messages: conversation.transcript(),
				});
				return;
			}
			if (method === "POST" && segments[3] === "messages" && segments.length === 4) {
				await this.handleMessage(conversation, request, response);
				return;
			}
			if (method === "PATCH" && segments[3] === "model" && segments.length === 4) {
				await this.handleModelChange(conversation, request, response);
				return;
			}
		}
		if (method === "GET" && url.pathname === "/v1/learning/events") {
			const limitParam = url.searchParams.get("limit");
			const limit = limitParam ? Number.parseInt(limitParam, 10) : 100;
			if (!Number.isFinite(limit) || limit <= 0 || limit > 1000) {
				throw new HttpError(400, "limit must be an integer in [1, 1000]");
			}
			const events = await this.deps.ledger.list({
				limit,
				conversationId: url.searchParams.get("conversation_id") ?? undefined,
			});
			sendJson(response, 200, { events });
			return;
		}
		if (method === "POST" && url.pathname === "/v1/learning/revert") {
			await this.handleRevert(request, response);
			return;
		}

		throw new HttpError(404, `No route: ${method} ${url.pathname}`);
	}

	private async handleHealth(response: http.ServerResponse): Promise<void> {
		let backend: { ok: boolean; detail: string };
		try {
			const health = await this.deps.backend.health();
			backend = { ok: health.status === "ok" || health.status === "healthy", detail: health.status };
		} catch (error) {
			backend = { ok: false, detail: error instanceof Error ? error.message : String(error) };
		}
		sendJson(response, backend.ok ? 200 : 503, {
			seat: "ok",
			backend,
			conversations: this.conversations.size,
			mcp_servers: this.deps.mcpHost.listServers(),
		});
	}

	private async handleModels(url: URL, response: http.ServerResponse): Promise<void> {
		let localErrors: Record<string, string> = {};
		if (url.searchParams.get("refresh")) {
			localErrors = await this.deps.registry.refreshLocal();
		}
		const models = await this.deps.registry.listAvailable();
		sendJson(response, 200, { models, local_errors: localErrors });
	}

	private async handleCreateConversation(
		request: http.IncomingMessage,
		response: http.ServerResponse,
	): Promise<void> {
		const body = parseJson<CreateConversationBody>(await readBody(request));
		if (!body.user_id || typeof body.user_id !== "string") throw new HttpError(400, "user_id is required");
		if (!body.provider || typeof body.provider !== "string") throw new HttpError(400, "provider is required");
		if (!body.model || typeof body.model !== "string") throw new HttpError(400, "model is required");

		const model = this.deps.registry.resolve(body.provider, body.model);
		if (!model) throw new HttpError(400, `Unknown model: ${body.provider}/${body.model}`);

		const deps: ConversationDeps = {
			backend: this.deps.backend,
			registry: this.deps.registry,
			ledger: this.deps.ledger,
			mcpTools: this.deps.mcpHost.getTools(),
		};
		let conversation: Conversation;
		try {
			conversation = new Conversation(deps, {
				userId: body.user_id,
				model,
				systemPrompt: typeof body.system_prompt === "string" ? body.system_prompt : undefined,
			});
		} catch (error) {
			throw new HttpError(400, error instanceof Error ? error.message : String(error));
		}
		this.conversations.set(conversation.id, conversation);
		sendJson(response, 201, {
			conversation_id: conversation.id,
			user_id: conversation.userId,
			harness_user_id: conversation.harnessUserId,
			model: conversation.model,
		});
	}

	private async handleMessage(
		conversation: Conversation,
		request: http.IncomingMessage,
		response: http.ServerResponse,
	): Promise<void> {
		const body = parseJson<{ text?: string }>(await readBody(request));
		if (!body.text || typeof body.text !== "string" || !body.text.trim()) {
			throw new HttpError(400, "text is required");
		}
		if (conversation.isStreaming) throw new HttpError(409, "Conversation is busy");

		response.writeHead(200, {
			"Content-Type": "text/event-stream; charset=utf-8",
			"Cache-Control": "no-cache, no-transform",
			Connection: "keep-alive",
			"X-Accel-Buffering": "no",
		});
		response.write("retry: 5000\n\n");

		const heartbeat = setInterval(() => {
			if (!response.writableEnded) response.write(": ping\n\n");
		}, SSE_HEARTBEAT_MS);

		// A destroyed response must never take the agent loop down: writes are
		// guarded and stream errors are absorbed (the disconnect handler below
		// aborts the run).
		response.on("error", () => {});
		const sink = (event: SeatEvent): void => {
			if (response.writableEnded || response.destroyed) return;
			try {
				response.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
			} catch {
				// Socket torn down mid-write; the close handler aborts the run.
			}
		};

		let clientGone = false;
		request.on("close", () => {
			if (!response.writableEnded) {
				clientGone = true;
				conversation.abort();
			}
		});

		try {
			await conversation.sendMessage(body.text, sink);
		} catch (error) {
			if (error instanceof ConversationBusyError) {
				sink({ type: "error", message: error.message });
			} else if (!clientGone) {
				sink({ type: "error", message: error instanceof Error ? error.message : String(error) });
			}
		} finally {
			clearInterval(heartbeat);
			if (!response.writableEnded) response.end();
		}
	}

	private async handleModelChange(
		conversation: Conversation,
		request: http.IncomingMessage,
		response: http.ServerResponse,
	): Promise<void> {
		const body = parseJson<{ provider?: string; model?: string }>(await readBody(request));
		if (!body.provider || !body.model) throw new HttpError(400, "provider and model are required");
		try {
			const model = conversation.setModel(body.provider, body.model);
			sendJson(response, 200, { model });
		} catch (error) {
			if (error instanceof UnknownModelError) throw new HttpError(400, error.message);
			if (error instanceof ConversationBusyError) throw new HttpError(409, error.message);
			throw error;
		}
	}

	private async handleRevert(request: http.IncomingMessage, response: http.ServerResponse): Promise<void> {
		const body = parseJson<{ event_id?: string }>(await readBody(request));
		if (!body.event_id || typeof body.event_id !== "string") throw new HttpError(400, "event_id is required");
		try {
			const revertEntry = await this.deps.ledger.revert(body.event_id, this.deps.backend);
			sendJson(response, 200, { revert: revertEntry });
		} catch (error) {
			if (error instanceof LedgerError) throw new HttpError(409, error.message);
			throw error;
		}
	}
}
