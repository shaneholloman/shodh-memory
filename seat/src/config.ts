/**
 * Seat harness configuration, resolved from environment variables.
 *
 * Backend resolution mirrors mcp-server/index.ts:
 *   SHODH_API_URL > SHODH_HOST+SHODH_PORT > http://127.0.0.1:3030
 * API key resolution mirrors mcp-server/index.ts:
 *   SHODH_API_KEY > SHODH_DEV_API_KEY > first entry of SHODH_API_KEYS
 * (No auto-generation here: unlike the MCP server, the seat does not spawn the
 * backend, so a generated key would never match a running server.)
 */

import * as os from "node:os";
import * as path from "node:path";

export interface McpServerConfig {
	/** Short identifier used in tool-name prefixes: [a-zA-Z0-9_-]+ */
	name: string;
	command: string;
	args?: string[];
	env?: Record<string, string>;
	cwd?: string;
}

export interface SeatConfig {
	/** shodh-memory Rust backend base URL (no trailing slash). */
	apiUrl: string;
	/** X-API-Key value for the backend. */
	apiKey: string;
	/** Bind host of the seat HTTP server. */
	host: string;
	/** Bind port of the seat HTTP server. */
	port: number;
	/** Bearer token required on seat endpoints. Mandatory for non-loopback binds. */
	authToken?: string;
	/** Directory for the learning ledger and other durable seat state. */
	dataDir: string;
	/** OpenAI-compatible base URL for Ollama (its /v1 endpoint). */
	ollamaBaseUrl: string;
	/** OpenAI-compatible base URL for LM Studio. */
	lmStudioBaseUrl: string;
	/** OpenAI-compatible base URL for vLLM (`vllm serve` mounts OpenAI routes
	 *  under /v1 and defaults to port 8000). */
	vllmBaseUrl: string;
	/** Context window advertised for locally served models (they do not report one). */
	localContextWindow: number;
	/** Max output tokens for locally served models. */
	localMaxTokens: number;
	/** Optional path to a JSON file: { "servers": McpServerConfig[] } */
	mcpConfigPath?: string;
	/** Backend request timeout in milliseconds. */
	backendTimeoutMs: number;
}

function isLoopback(host: string): boolean {
	return host === "127.0.0.1" || host === "localhost" || host === "::1";
}

function resolveApiUrl(env: NodeJS.ProcessEnv): string {
	if (env.SHODH_API_URL) return env.SHODH_API_URL.replace(/\/+$/, "");
	const host = env.SHODH_HOST;
	const port = env.SHODH_PORT;
	if (host) {
		const scheme = port === "443" ? "https" : "http";
		const portSuffix = port && port !== "443" && port !== "80" ? `:${port}` : "";
		return `${scheme}://${host}${portSuffix}`;
	}
	if (port) return `http://127.0.0.1:${port}`;
	return "http://127.0.0.1:3030";
}

function resolveApiKey(env: NodeJS.ProcessEnv): string {
	if (env.SHODH_API_KEY) return env.SHODH_API_KEY;
	if (env.SHODH_DEV_API_KEY) return env.SHODH_DEV_API_KEY;
	const first = env.SHODH_API_KEYS?.split(",")[0]?.trim();
	if (first) return first;
	return "";
}

/**
 * Default data directory. Deliberately NOT under the repository or any
 * cloud-synced/user-watched folder: file watchers have been observed to
 * corrupt append-heavy stores on this project (see
 * finding-bm25-onedrive-silent-commit-loss). LOCALAPPDATA on Windows,
 * XDG data dir elsewhere.
 */
function defaultDataDir(env: NodeJS.ProcessEnv): string {
	if (process.platform === "win32") {
		const base = env.LOCALAPPDATA ?? path.join(os.homedir(), "AppData", "Local");
		return path.join(base, "shodh", "seat-harness");
	}
	const base = env.XDG_DATA_HOME ?? path.join(os.homedir(), ".local", "share");
	return path.join(base, "shodh", "seat-harness");
}

function parseIntEnv(value: string | undefined, fallback: number, name: string): number {
	if (value === undefined || value === "") return fallback;
	const parsed = Number.parseInt(value, 10);
	if (!Number.isFinite(parsed) || parsed <= 0) {
		throw new Error(`Invalid ${name}: "${value}" (expected a positive integer)`);
	}
	return parsed;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): SeatConfig {
	const apiKey = resolveApiKey(env);
	if (!apiKey) {
		throw new Error(
			"No backend API key configured. Set SHODH_API_KEY (or SHODH_DEV_API_KEY / SHODH_API_KEYS) " +
				"to the key the shodh-memory server accepts.",
		);
	}

	const host = env.SEAT_HOST ?? "127.0.0.1";
	const authToken = env.SEAT_AUTH_TOKEN?.trim() || undefined;
	if (!isLoopback(host) && !authToken) {
		throw new Error(
			`SEAT_HOST=${host} is not loopback; refusing to start without SEAT_AUTH_TOKEN. ` +
				"Provider credentials live in this process — never expose it unauthenticated.",
		);
	}

	return {
		apiUrl: resolveApiUrl(env),
		apiKey,
		host,
		port: parseIntEnv(env.SEAT_PORT, 3141, "SEAT_PORT"),
		authToken,
		dataDir: env.SEAT_DATA_DIR ?? defaultDataDir(env),
		ollamaBaseUrl: (env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434/v1").replace(/\/+$/, ""),
		lmStudioBaseUrl: (env.LMSTUDIO_BASE_URL ?? "http://127.0.0.1:1234/v1").replace(/\/+$/, ""),
		vllmBaseUrl: (env.VLLM_BASE_URL ?? "http://127.0.0.1:8000/v1").replace(/\/+$/, ""),
		localContextWindow: parseIntEnv(env.SEAT_LOCAL_CONTEXT_WINDOW, 32768, "SEAT_LOCAL_CONTEXT_WINDOW"),
		localMaxTokens: parseIntEnv(env.SEAT_LOCAL_MAX_TOKENS, 8192, "SEAT_LOCAL_MAX_TOKENS"),
		mcpConfigPath: env.SEAT_MCP_SERVERS,
		backendTimeoutMs: parseIntEnv(env.SEAT_BACKEND_TIMEOUT_MS, 30000, "SEAT_BACKEND_TIMEOUT_MS"),
	};
}
