/**
 * Structured events streamed to the client over SSE.
 *
 * The traceability surface is the product: memory operations are never opaque.
 * Every recall carries ids, scores, and full ScoreAttribution; every learning
 * update (write or reinforcement) carries its ledger event id so the UI can
 * review and revert it.
 */

import type {
	RecallFact,
	RecallLineageEdge,
	RecallMemory,
	RecallTodo,
	ReinforceOutcome,
	ReinforceStats,
} from "./backend.js";

/** Which memory namespace an operation touched. */
export type MemoryScope = "user" | "harness";

export interface ModelRef {
	provider: string;
	id: string;
	name: string;
}

export interface UsagePayload {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	reasoning?: number;
	totalTokens: number;
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
		total: number;
	};
}

export type ReinforceTrigger =
	| { kind: "response_overlap"; overlaps: Record<string, number>; threshold: number }
	| { kind: "citation"; cited: string[] }
	| { kind: "negative_followup"; keywords: string[] }
	| { kind: "revert"; of: string };

export type SeatEvent =
	| { type: "conversation_created"; conversation_id: string; user_id: string; model: ModelRef }
	| { type: "turn_start"; turn: number }
	| { type: "text_delta"; delta: string }
	| { type: "thinking_delta"; delta: string }
	| { type: "tool_call_start"; tool_call_id: string; tool_name: string; args: unknown }
	| { type: "tool_call_end"; tool_call_id: string; tool_name: string; is_error: boolean }
	| {
			type: "memory_recall";
			scope: MemoryScope;
			tool_call_id?: string;
			query: string;
			mode: string;
			memories: RecallMemory[];
			facts: RecallFact[];
			todos: RecallTodo[];
			lineage: RecallLineageEdge[];
			took_ms: number;
	  }
	| {
			type: "memory_write";
			scope: MemoryScope;
			memory_id: string;
			memory_type: string;
			content_preview: string;
			ledger_event_id: string;
	  }
	| {
			type: "memory_reinforce";
			scope: MemoryScope;
			outcome: ReinforceOutcome;
			memory_ids: string[];
			stats: ReinforceStats;
			trigger: ReinforceTrigger;
			ledger_event_id: string;
	  }
	| { type: "harness_learning_applied"; memories: { id: string; content: string; score: number }[] }
	| { type: "model_changed"; model: ModelRef }
	| { type: "usage"; model: ModelRef; usage: UsagePayload }
	| { type: "turn_end"; turn: number; stop_reason: string; error_message?: string }
	| { type: "agent_end" }
	| { type: "error"; message: string };

export type SeatEventSink = (event: SeatEvent) => void;
