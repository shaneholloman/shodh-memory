/**
 * Learning ledger: every update the learning loops make to shodh-memory state
 * is recorded as an append-only, reviewable, revertible event BEFORE the UI or
 * anyone else has to ask "what changed and why".
 *
 * Design:
 * - Append-only JSONL. Reverts are themselves appended events referencing the
 *   original (`kind: "revert"`, data.of = original id); nothing is mutated.
 * - Revert semantics are honest about what the backend supports:
 *   - memory writes revert exactly (DELETE /api/memory/{id}).
 *   - "helpful"/"misleading" reinforcements revert by applying the opposite
 *     outcome through the same /api/reinforce path. The backend's momentum
 *     update (EMA with inertia, src/memory/feedback.rs) is not exactly
 *     invertible, so this is a compensating action, not a bitwise undo —
 *     recorded as such in the revert event.
 *   - "neutral" reinforcements record access only; there is nothing to
 *     compensate, and the revert event says so.
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as fsp from "node:fs/promises";
import * as path from "node:path";
import type { ReinforceOutcome, ReinforceStats, ShodhBackend } from "./backend.js";
import type { MemoryScope, ReinforceTrigger } from "./events.js";

export interface MemoryWriteData {
	memory_id: string;
	memory_type: string;
	content_preview: string;
	trigger: "model_tool_call" | "empty_recall_capture" | "tool_error_capture";
}

export interface ReinforceData {
	outcome: ReinforceOutcome;
	memory_ids: string[];
	trigger: ReinforceTrigger;
	stats: ReinforceStats;
}

export interface RevertData {
	of: string;
	compensation:
		| { kind: "memory_delete"; memory_id: string }
		| { kind: "counter_reinforce"; outcome: ReinforceOutcome; memory_ids: string[]; stats: ReinforceStats }
		| { kind: "none" };
	note: string;
}

export type LedgerEntry =
	| LedgerEntryBase<"memory_write", MemoryWriteData>
	| LedgerEntryBase<"reinforce", ReinforceData>
	| LedgerEntryBase<"revert", RevertData>;

interface LedgerEntryBase<K extends string, D> {
	id: string;
	ts: string;
	kind: K;
	scope: MemoryScope;
	/** The actual backend user_id the operation ran against (harness scope uses the derived namespace). */
	user_id: string;
	conversation_id: string;
	turn: number;
	data: D;
}

export interface LedgerEntryView {
	entry: LedgerEntry;
	reverted_by?: string;
}

export class LedgerError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "LedgerError";
	}
}

export class LearningLedger {
	private readonly filePath: string;
	/** Serializes appends so entries never interleave. */
	private writeChain: Promise<void> = Promise.resolve();

	constructor(dataDir: string) {
		fs.mkdirSync(dataDir, { recursive: true });
		this.filePath = path.join(dataDir, "learning-ledger.jsonl");
	}

	get file(): string {
		return this.filePath;
	}

	async append<K extends LedgerEntry["kind"]>(
		kind: K,
		scope: MemoryScope,
		userId: string,
		conversationId: string,
		turn: number,
		data: Extract<LedgerEntry, { kind: K }>["data"],
	): Promise<LedgerEntry> {
		const entry = {
			id: crypto.randomUUID(),
			ts: new Date().toISOString(),
			kind,
			scope,
			user_id: userId,
			conversation_id: conversationId,
			turn,
			data,
		} as LedgerEntry;

		const write = this.writeChain.then(() => fsp.appendFile(this.filePath, `${JSON.stringify(entry)}\n`, "utf8"));
		// Keep the chain alive even if a write fails; the failure surfaces to the caller.
		this.writeChain = write.catch(() => {});
		await write;
		return entry;
	}

	private async readAll(): Promise<LedgerEntry[]> {
		let raw: string;
		try {
			raw = await fsp.readFile(this.filePath, "utf8");
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
			throw error;
		}
		const entries: LedgerEntry[] = [];
		for (const line of raw.split("\n")) {
			const trimmed = line.trim();
			if (!trimmed) continue;
			try {
				entries.push(JSON.parse(trimmed) as LedgerEntry);
			} catch {
				// A torn trailing line (crash mid-append) is skipped; everything
				// before it is intact because appends are serialized.
			}
		}
		return entries;
	}

	async list(options: { limit?: number; conversationId?: string } = {}): Promise<LedgerEntryView[]> {
		const limit = options.limit ?? 100;
		const entries = await this.readAll();
		const revertedBy = new Map<string, string>();
		for (const entry of entries) {
			if (entry.kind === "revert") revertedBy.set(entry.data.of, entry.id);
		}
		const filtered = options.conversationId
			? entries.filter((entry) => entry.conversation_id === options.conversationId)
			: entries;
		return filtered
			.slice(-limit)
			.reverse()
			.map((entry) => ({
				entry,
				reverted_by: revertedBy.get(entry.id),
			}));
	}

	async get(id: string): Promise<LedgerEntryView | undefined> {
		const entries = await this.readAll();
		const entry = entries.find((candidate) => candidate.id === id);
		if (!entry) return undefined;
		const revert = entries.find(
			(candidate) => candidate.kind === "revert" && candidate.data.of === id,
		);
		return { entry, reverted_by: revert?.id };
	}

	/**
	 * Revert a learning event by applying its compensating action through the
	 * backend, then recording the revert as a new ledger event.
	 */
	async revert(id: string, backend: ShodhBackend): Promise<LedgerEntry> {
		const view = await this.get(id);
		if (!view) throw new LedgerError(`Unknown ledger event: ${id}`);
		if (view.reverted_by) throw new LedgerError(`Event ${id} was already reverted by ${view.reverted_by}`);
		const original = view.entry;
		if (original.kind === "revert") throw new LedgerError("Revert events cannot be reverted");

		let compensation: RevertData["compensation"];
		let note: string;

		if (original.kind === "memory_write") {
			await backend.deleteMemory(original.user_id, original.data.memory_id);
			compensation = { kind: "memory_delete", memory_id: original.data.memory_id };
			note = "Exact revert: the written memory was deleted.";
		} else {
			const outcome = original.data.outcome;
			if (outcome === "neutral") {
				compensation = { kind: "none" };
				note = "Neutral reinforcement records access only; no compensating action exists.";
			} else {
				const inverse: ReinforceOutcome = outcome === "helpful" ? "misleading" : "helpful";
				const stats = await backend.reinforce(original.user_id, original.data.memory_ids, inverse);
				compensation = { kind: "counter_reinforce", outcome: inverse, memory_ids: original.data.memory_ids, stats };
				note =
					"Compensating action: opposite outcome applied via /api/reinforce. " +
					"The backend momentum update (EMA with inertia) is not exactly invertible.";
			}
		}

		return this.append("revert", original.scope, original.user_id, original.conversation_id, original.turn, {
			of: original.id,
			compensation,
			note,
		});
	}
}
