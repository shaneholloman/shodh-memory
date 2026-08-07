/**
 * A conversation: one pi Agent wired to shodh-memory, with two learning loops
 * closing at every turn.
 *
 * Loop 1 — memory-level (user scope): memories recalled during a turn are
 * reinforced through the backend's existing Hebbian path (POST /api/reinforce)
 * according to whether the response actually used them (citation or token
 * overlap, mirroring src/memory/feedback.rs semantics). A negative follow-up
 * from the user penalizes the previous turn's surfaced memories.
 *
 * Loop 2 — harness-level (harness scope): operational lessons about retrieval
 * and tool use are stored AS MEMORIES in an isolated namespace
 * (`<user_id>.seat-harness`), surfaced by the same recall machinery before
 * each turn, injected as a labeled system-prompt block, and reinforced by the
 * same rules. One substrate, two scopes; the scopes never share retrieval
 * because the backend keys every store (RocksDB, graph, feedback) by user_id
 * (src/handlers/state.rs get_user_memory / get_user_graph).
 *
 * Every update either loop makes is recorded in the LearningLedger before the
 * conversation continues — reviewable and revertible from the start.
 */

import * as crypto from "node:crypto";
import { Agent, type AgentEvent, type AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import type { RecallMemory, ReinforceOutcome, ShodhBackend } from "./backend.js";
import type { MemoryScope, ModelRef, ReinforceTrigger, SeatEvent, SeatEventSink, UsagePayload } from "./events.js";
import {
	detectNegativeKeywords,
	extractCitations,
	extractTokens,
	memoryOverlap,
	OVERLAP_USED_THRESHOLD,
} from "./feedback.js";
import type { LearningLedger } from "./ledger.js";
import { createMemoryTools } from "./memory-tools.js";
import type { ModelRegistry } from "./models-registry.js";

const HARNESS_SUFFIX = ".seat-harness";
/** Backend limit: src/validation.rs MAX_USER_ID_LENGTH = 128. */
const MAX_USER_ID_LENGTH = 128;
const USER_ID_PATTERN = /^[A-Za-z0-9@._-]+$/;
/** Minimum normalized recall score for a harness learning to be injected. */
const HARNESS_INJECT_MIN_SCORE = 0.25;
const HARNESS_INJECT_LIMIT = 3;
/** Caps on automatic harness captures, per conversation. */
const MAX_EMPTY_RECALL_CAPTURES = 5;
const MAX_TOOL_ERROR_CAPTURES = 5;

const BASE_SYSTEM_PROMPT = `You are the shodh-memory conversation seat: an assistant whose persistent memory is visible and inspectable by the user.

Memory discipline:
- Use recall_memory when the user refers to past work, decisions, people, or preferences, or when prior context would materially improve the answer.
- When a recalled memory informs your answer, cite it inline as [mem:<id>] using the id shown in the recall result.
- Use remember_memory sparingly: durable facts, decisions, and learnings only.
- Use record_seat_learning only for operational lessons about retrieval or tool strategy — never for user content.`;

export class ConversationBusyError extends Error {
	constructor() {
		super("Conversation is currently processing a message");
		this.name = "ConversationBusyError";
	}
}

export class UnknownModelError extends Error {
	constructor(provider: string, id: string) {
		super(`Unknown or unavailable model: ${provider}/${id}`);
		this.name = "UnknownModelError";
	}
}

export interface ConversationDeps {
	backend: ShodhBackend;
	registry: ModelRegistry;
	ledger: LearningLedger;
	mcpTools: AgentTool<any>[];
}

export interface ConversationOptions {
	userId: string;
	model: Model<Api>;
	systemPrompt?: string;
}

interface SurfacedMemory {
	scope: MemoryScope;
	content: string;
}

export function deriveHarnessUserId(userId: string): string {
	const derived = `${userId}${HARNESS_SUFFIX}`;
	if (!USER_ID_PATTERN.test(userId) || userId.includes("..") || userId.startsWith(".")) {
		throw new Error(`Invalid user_id "${userId}" (allowed: alphanumeric, -, _, @, .)`);
	}
	if (derived.length > MAX_USER_ID_LENGTH) {
		throw new Error(`user_id too long: harness namespace "${derived}" exceeds ${MAX_USER_ID_LENGTH} chars`);
	}
	return derived;
}

function modelRef(model: Model<Api>): ModelRef {
	return { provider: model.provider, id: model.id, name: model.name };
}

function usagePayload(usage: {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	reasoning?: number;
	totalTokens: number;
	cost: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number };
}): UsagePayload {
	return {
		input: usage.input,
		output: usage.output,
		cacheRead: usage.cacheRead,
		cacheWrite: usage.cacheWrite,
		reasoning: usage.reasoning,
		totalTokens: usage.totalTokens,
		cost: { ...usage.cost },
	};
}

function memoryShortId(memoryId: string): string {
	return memoryId.replace(/-/g, "").slice(0, 8).toLowerCase();
}

export class Conversation {
	readonly id: string;
	readonly userId: string;
	readonly harnessUserId: string;
	readonly createdAt: Date;

	private readonly deps: ConversationDeps;
	private readonly agent: Agent;
	private readonly baseSystemPrompt: string;

	private turn = 0;
	private currentSink?: SeatEventSink;
	/** Events raised outside an active run (e.g. model change), flushed on the next run. */
	private pendingEvents: SeatEvent[] = [];

	/** Memories surfaced during the current run, keyed by memory id. */
	private surfaced = new Map<string, SurfacedMemory>();
	/** Memories surfaced during the previous run (target of negative-followup penalties). */
	private prevSurfaced = new Map<string, SurfacedMemory>();
	private emptyRecallQueries: string[] = [];
	private toolErrors: { toolName: string; message: string }[] = [];
	private assistantTexts: string[] = [];
	private lastStopReason = "stop";
	private lastErrorMessage: string | undefined;

	/** Per-conversation dedupe for automatic harness captures. */
	private capturedEmptyRecalls = new Set<string>();
	private capturedToolErrors = new Set<string>();

	constructor(deps: ConversationDeps, options: ConversationOptions) {
		this.id = crypto.randomUUID();
		this.userId = options.userId;
		this.harnessUserId = deriveHarnessUserId(options.userId);
		this.createdAt = new Date();
		this.deps = deps;
		this.baseSystemPrompt = options.systemPrompt?.trim()
			? `${BASE_SYSTEM_PROMPT}\n\n${options.systemPrompt.trim()}`
			: BASE_SYSTEM_PROMPT;

		const memoryTools = createMemoryTools({
			backend: deps.backend,
			userId: this.userId,
			harnessUserId: this.harnessUserId,
			conversationId: this.id,
			getTurn: () => this.turn,
			emit: (event) => this.emit(event),
			onSurfaced: (scope, memories) => {
				for (const memory of memories) {
					this.surfaced.set(memory.id, { scope, content: memory.content });
				}
			},
			onEmptyRecall: (query) => {
				this.emptyRecallQueries.push(query);
			},
			ledger: deps.ledger,
		});

		this.agent = new Agent({
			initialState: {
				systemPrompt: this.baseSystemPrompt,
				model: options.model,
				thinkingLevel: "off",
				tools: [...memoryTools, ...deps.mcpTools],
			},
			streamFn: (model, context, streamOptions) => deps.registry.models.streamSimple(model, context, streamOptions),
		});
		this.agent.subscribe((event) => this.onAgentEvent(event));
	}

	get model(): ModelRef {
		return modelRef(this.agent.state.model);
	}

	get isStreaming(): boolean {
		return this.agent.state.isStreaming;
	}

	private emit(event: SeatEvent): void {
		if (this.currentSink) {
			this.currentSink(event);
		} else {
			this.pendingEvents.push(event);
		}
	}

	private onAgentEvent(event: AgentEvent): void {
		switch (event.type) {
			case "message_update": {
				const streamEvent = event.assistantMessageEvent;
				if (streamEvent.type === "text_delta") {
					this.emit({ type: "text_delta", delta: streamEvent.delta });
				} else if (streamEvent.type === "thinking_delta") {
					this.emit({ type: "thinking_delta", delta: streamEvent.delta });
				}
				break;
			}
			case "message_end": {
				const message = event.message;
				if (typeof message === "object" && message !== null && "role" in message && message.role === "assistant") {
					this.lastStopReason = message.stopReason;
					this.lastErrorMessage = message.errorMessage;
					const text = message.content
						.filter((block): block is { type: "text"; text: string } => block.type === "text")
						.map((block) => block.text)
						.join("");
					if (text) this.assistantTexts.push(text);
					this.emit({ type: "usage", model: this.model, usage: usagePayload(message.usage) });
				}
				break;
			}
			case "tool_execution_start":
				this.emit({
					type: "tool_call_start",
					tool_call_id: event.toolCallId,
					tool_name: event.toolName,
					args: event.args,
				});
				break;
			case "tool_execution_end": {
				this.emit({
					type: "tool_call_end",
					tool_call_id: event.toolCallId,
					tool_name: event.toolName,
					is_error: event.isError,
				});
				if (event.isError) {
					const message =
						typeof event.result === "string" ? event.result : JSON.stringify(event.result)?.slice(0, 500);
					this.toolErrors.push({ toolName: event.toolName, message: message ?? "unknown error" });
				}
				break;
			}
			default:
				break;
		}
	}

	/**
	 * Run one user message through the agent, streaming SeatEvents to `sink`.
	 * Resolves after the run AND the learning loops have completed.
	 */
	async sendMessage(text: string, sink: SeatEventSink): Promise<void> {
		if (this.agent.state.isStreaming || this.currentSink) throw new ConversationBusyError();
		this.currentSink = sink;
		this.turn += 1;

		// Reset per-run state.
		this.surfaced = new Map();
		this.emptyRecallQueries = [];
		this.toolErrors = [];
		this.assistantTexts = [];
		this.lastStopReason = "stop";
		this.lastErrorMessage = undefined;

		try {
			for (const pending of this.pendingEvents) sink(pending);
			this.pendingEvents = [];

			this.emit({ type: "turn_start", turn: this.turn });

			await this.applyNegativeFollowupPenalty(text);
			await this.injectHarnessLearnings(text);

			await this.agent.prompt(text);

			await this.closeLearningLoops();

			this.emit({
				type: "turn_end",
				turn: this.turn,
				stop_reason: this.lastStopReason,
				error_message: this.lastErrorMessage,
			});
			this.emit({ type: "agent_end" });
		} finally {
			this.prevSurfaced = this.surfaced;
			this.currentSink = undefined;
		}
	}

	/**
	 * Mirror of the backend's user_followup penalty (src/memory/feedback.rs):
	 * a correction/frustration message penalizes the memories surfaced on the
	 * previous turn.
	 */
	private async applyNegativeFollowupPenalty(userText: string): Promise<void> {
		if (this.prevSurfaced.size === 0) return;
		const keywords = detectNegativeKeywords(userText);
		if (keywords.length === 0) return;

		const byScope = new Map<MemoryScope, string[]>();
		for (const [memoryId, memory] of this.prevSurfaced) {
			const ids = byScope.get(memory.scope) ?? [];
			ids.push(memoryId);
			byScope.set(memory.scope, ids);
		}
		for (const [scope, ids] of byScope) {
			await this.reinforceAndRecord(scope, ids, "misleading", {
				kind: "negative_followup",
				keywords,
			});
		}
	}

	/**
	 * Harness-level learning, read side: recall operational lessons from the
	 * harness scope with the user message as cue and inject strong matches as
	 * a labeled system-prompt block for this run only.
	 */
	private async injectHarnessLearnings(userText: string): Promise<void> {
		let memories: RecallMemory[] = [];
		try {
			const startedAt = Date.now();
			const response = await this.deps.backend.recall({
				userId: this.harnessUserId,
				query: userText,
				limit: HARNESS_INJECT_LIMIT,
				mode: "hybrid",
				debug: true,
			});
			memories = response.memories.filter((memory) => memory.score >= HARNESS_INJECT_MIN_SCORE);
			if (memories.length > 0) {
				this.emit({
					type: "memory_recall",
					scope: "harness",
					query: userText,
					mode: "hybrid",
					memories,
					facts: [],
					todos: [],
					lineage: [],
					took_ms: Date.now() - startedAt,
				});
			}
		} catch (error) {
			// Harness recall is an enhancement; its failure must not block the turn.
			this.emit({
				type: "error",
				message: `Harness-scope recall failed: ${error instanceof Error ? error.message : String(error)}`,
			});
		}

		if (memories.length === 0) {
			this.agent.state.systemPrompt = this.baseSystemPrompt;
			return;
		}

		for (const memory of memories) {
			this.surfaced.set(memory.id, { scope: "harness", content: memory.experience.content });
		}
		const lines = memories.map((memory) => `- ${memory.experience.content}`);
		this.agent.state.systemPrompt = `${this.baseSystemPrompt}\n\n## Learned operating notes (from previous sessions of this assistant)\n${lines.join("\n")}`;
		this.emit({
			type: "harness_learning_applied",
			memories: memories.map((memory) => ({
				id: memory.id,
				content: memory.experience.content,
				score: memory.score,
			})),
		});
	}

	/** Reinforce + ledger + emit, with failures surfaced as error events. */
	private async reinforceAndRecord(
		scope: MemoryScope,
		memoryIds: string[],
		outcome: ReinforceOutcome,
		trigger: ReinforceTrigger,
	): Promise<void> {
		if (memoryIds.length === 0) return;
		const scopeUserId = scope === "user" ? this.userId : this.harnessUserId;
		try {
			const stats = await this.deps.backend.reinforce(scopeUserId, memoryIds, outcome);
			const ledgerEntry = await this.deps.ledger.append("reinforce", scope, scopeUserId, this.id, this.turn, {
				outcome,
				memory_ids: memoryIds,
				trigger,
				stats,
			});
			this.emit({
				type: "memory_reinforce",
				scope,
				outcome,
				memory_ids: memoryIds,
				stats,
				trigger,
				ledger_event_id: ledgerEntry.id,
			});
		} catch (error) {
			this.emit({
				type: "error",
				message: `Reinforcement (${outcome}) failed for ${scope} scope: ${
					error instanceof Error ? error.message : String(error)
				}`,
			});
		}
	}

	/**
	 * Close both learning loops for the finished run:
	 * 1. Reinforce surfaced memories by usage (citation or token overlap).
	 * 2. Capture deterministic harness learnings (empty recalls, tool errors).
	 */
	private async closeLearningLoops(): Promise<void> {
		const responseText = this.assistantTexts.join("\n");
		if (this.surfaced.size > 0 && responseText.length > 0) {
			const responseTokens = extractTokens(responseText);
			const citations = extractCitations(responseText);

			const groups = new Map<
				string,
				{ scope: MemoryScope; outcome: ReinforceOutcome; ids: string[]; overlaps: Record<string, number>; cited: string[] }
			>();
			for (const [memoryId, memory] of this.surfaced) {
				const cited = citations.has(memoryShortId(memoryId));
				const overlap = memoryOverlap(memory.content, responseTokens);
				const outcome: ReinforceOutcome = cited || overlap >= OVERLAP_USED_THRESHOLD ? "helpful" : "neutral";
				const key = `${memory.scope}:${outcome}`;
				const group =
					groups.get(key) ??
					({ scope: memory.scope, outcome, ids: [], overlaps: {}, cited: [] } as {
						scope: MemoryScope;
						outcome: ReinforceOutcome;
						ids: string[];
						overlaps: Record<string, number>;
						cited: string[];
					});
				group.ids.push(memoryId);
				group.overlaps[memoryId] = Number(overlap.toFixed(4));
				if (cited) group.cited.push(memoryId);
				groups.set(key, group);
			}

			for (const group of groups.values()) {
				const trigger =
					group.cited.length > 0
						? ({ kind: "citation", cited: group.cited } as const)
						: ({ kind: "response_overlap", overlaps: group.overlaps, threshold: OVERLAP_USED_THRESHOLD } as const);
				await this.reinforceAndRecord(group.scope, group.ids, group.outcome, trigger);
			}
		}

		await this.captureHarnessLearnings();
	}

	/** Deterministic write side of the harness loop, with per-conversation dedupe and caps. */
	private async captureHarnessLearnings(): Promise<void> {
		for (const query of this.emptyRecallQueries) {
			if (this.capturedEmptyRecalls.size >= MAX_EMPTY_RECALL_CAPTURES) break;
			const normalized = query.trim().toLowerCase();
			if (this.capturedEmptyRecalls.has(normalized)) continue;
			this.capturedEmptyRecalls.add(normalized);
			await this.writeHarnessCapture(
				`Recall returned no results for cue "${query.slice(0, 200)}". Rephrase with concrete entity names or broaden the cue before answering without memory.`,
				"learning",
				["seat-harness", "retrieval", "empty-recall"],
				"empty_recall_capture",
			);
		}

		for (const toolError of this.toolErrors) {
			if (this.capturedToolErrors.size >= MAX_TOOL_ERROR_CAPTURES) break;
			if (this.capturedToolErrors.has(toolError.toolName)) continue;
			this.capturedToolErrors.add(toolError.toolName);
			await this.writeHarnessCapture(
				`Tool ${toolError.toolName} failed: ${toolError.message.slice(0, 300)}. Verify arguments and tool availability before relying on it.`,
				"error",
				["seat-harness", "tool-error", toolError.toolName],
				"tool_error_capture",
			);
		}
	}

	private async writeHarnessCapture(
		content: string,
		memoryType: "learning" | "error",
		tags: string[],
		trigger: "empty_recall_capture" | "tool_error_capture",
	): Promise<void> {
		try {
			const response = await this.deps.backend.remember({
				userId: this.harnessUserId,
				content,
				memoryType,
				tags,
			});
			const ledgerEntry = await this.deps.ledger.append(
				"memory_write",
				"harness",
				this.harnessUserId,
				this.id,
				this.turn,
				{
					memory_id: response.id,
					memory_type: memoryType,
					content_preview: content.slice(0, 200),
					trigger,
				},
			);
			this.emit({
				type: "memory_write",
				scope: "harness",
				memory_id: response.id,
				memory_type: memoryType,
				content_preview: content.slice(0, 200),
				ledger_event_id: ledgerEntry.id,
			});
		} catch (error) {
			this.emit({
				type: "error",
				message: `Harness learning capture failed: ${error instanceof Error ? error.message : String(error)}`,
			});
		}
	}

	/**
	 * Swap the model for future turns. The transcript and all retrieved
	 * evidence stay exactly as they are — only the model producing the next
	 * turn changes.
	 */
	setModel(provider: string, id: string): ModelRef {
		if (this.agent.state.isStreaming) throw new ConversationBusyError();
		const model = this.deps.registry.resolve(provider, id);
		if (!model) throw new UnknownModelError(provider, id);
		this.agent.state.model = model;
		const ref = modelRef(model);
		this.emit({ type: "model_changed", model: ref });
		return ref;
	}

	abort(): void {
		this.agent.abort();
	}

	transcript(): unknown[] {
		return this.agent.state.messages.map((message) => message as unknown);
	}
}
