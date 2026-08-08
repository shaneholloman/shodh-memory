/**
 * Seat harness entry point.
 */

import * as fsp from "node:fs/promises";
import { ShodhBackend } from "./backend.js";
import { loadConfig, type McpServerConfig } from "./config.js";
import { FileCredentialStore } from "./credentials.js";
import { LearningLedger } from "./ledger.js";
import { McpHost } from "./mcp.js";
import { ModelRegistry } from "./models-registry.js";
import { SeatServer } from "./server.js";
import { SeatStore } from "./store.js";

async function loadMcpServers(configPath: string | undefined): Promise<McpServerConfig[]> {
	if (!configPath) return [];
	const raw = await fsp.readFile(configPath, "utf8");
	const parsed = JSON.parse(raw) as { servers?: McpServerConfig[] };
	if (!Array.isArray(parsed.servers)) {
		throw new Error(`${configPath}: expected { "servers": [ { name, command, args?, env?, cwd? } ] }`);
	}
	for (const server of parsed.servers) {
		if (!server.name || !server.command) {
			throw new Error(`${configPath}: every server needs "name" and "command"`);
		}
	}
	return parsed.servers;
}

async function main(): Promise<void> {
	const config = loadConfig();
	const backend = new ShodhBackend(config.apiUrl, config.apiKey, config.backendTimeoutMs);
	const credentials = new FileCredentialStore(config.dataDir);
	const registry = new ModelRegistry(config, credentials);
	const ledger = new LearningLedger(config.dataDir);
	const store = new SeatStore(config.dataDir);
	const mcpHost = new McpHost();

	try {
		const health = await backend.health();
		console.log(`[seat] backend ${config.apiUrl} reachable (status: ${health.status})`);
	} catch (error) {
		console.warn(
			`[seat] WARNING: backend ${config.apiUrl} not reachable yet — memory tools will fail until it is up. ` +
				`(${error instanceof Error ? error.message : String(error)})`,
		);
	}

	const localErrors = await registry.refreshLocal();
	for (const [provider, message] of Object.entries(localErrors)) {
		console.log(`[seat] local provider ${provider} offline: ${message}`);
	}

	const mcpServers = await loadMcpServers(config.mcpConfigPath);
	if (mcpServers.length > 0) {
		const mcpErrors = await mcpHost.connect(mcpServers);
		for (const server of mcpHost.listServers()) {
			console.log(`[seat] MCP server "${server.name}": ${server.tool_count} tools bridged`);
		}
		for (const [name, message] of Object.entries(mcpErrors)) {
			console.warn(`[seat] MCP server "${name}" failed to connect: ${message}`);
		}
	}

	const server = new SeatServer({ config, backend, registry, ledger, mcpHost, store });
	await server.listen();
	console.log(`[seat] listening on http://${config.host}:${config.port}`);
	console.log(`[seat] learning ledger: ${ledger.file}`);
	console.log(`[seat] conversation store + provider credentials: ${config.dataDir}`);

	const shutdown = async (signal: string): Promise<void> => {
		console.log(`[seat] ${signal} received, shutting down`);
		await server.close();
		await mcpHost.close();
		process.exit(0);
	};
	process.on("SIGINT", () => void shutdown("SIGINT"));
	process.on("SIGTERM", () => void shutdown("SIGTERM"));
}

main().catch((error) => {
	console.error(`[seat] fatal: ${error instanceof Error ? error.message : String(error)}`);
	process.exit(1);
});
