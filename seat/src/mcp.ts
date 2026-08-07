/**
 * MCP → agent-loop bridge.
 *
 * pi has no MCP client (packages/agent/src contains none; pi's README states
 * "No MCP" explicitly), so this bridge is ours: it connects to MCP servers
 * over stdio and exposes their tools as pi `AgentTool`s.
 *
 * Schema note: MCP tools publish plain JSON Schema. pi's tool-argument
 * validation handles non-TypeBox schemas explicitly
 * (packages/ai/src/utils/validation.ts `validateToolArguments`, TYPEBOX_KIND
 * branch), so the inputSchema passes through unchanged.
 *
 * Tool names follow the mcp__<server>__<tool> convention used across this
 * repository's tooling.
 */

import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import type { TSchema } from "@earendil-works/pi-ai";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { getDefaultEnvironment, StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import type { McpServerConfig } from "./config.js";

interface ConnectedServer {
	name: string;
	client: Client;
	tools: AgentTool<any>[];
}

const NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;
const CALL_TIMEOUT_MS = 120_000;

function mapContent(content: unknown): (TextContent | ImageContent)[] {
	if (!Array.isArray(content)) return [{ type: "text", text: "" }];
	const mapped: (TextContent | ImageContent)[] = [];
	for (const block of content) {
		if (block && typeof block === "object" && "type" in block) {
			const typed = block as { type: string; text?: string; data?: string; mimeType?: string };
			if (typed.type === "text" && typeof typed.text === "string") {
				mapped.push({ type: "text", text: typed.text });
				continue;
			}
			if (typed.type === "image" && typeof typed.data === "string" && typeof typed.mimeType === "string") {
				mapped.push({ type: "image", data: typed.data, mimeType: typed.mimeType });
				continue;
			}
		}
		mapped.push({ type: "text", text: JSON.stringify(block) });
	}
	return mapped.length > 0 ? mapped : [{ type: "text", text: "" }];
}

function toAgentTool(serverName: string, client: Client, tool: {
	name: string;
	description?: string;
	inputSchema: unknown;
	title?: string;
}): AgentTool<any> {
	return {
		name: `mcp__${serverName}__${tool.name}`,
		label: tool.title ?? tool.name,
		description: tool.description ?? `${tool.name} (MCP tool from ${serverName})`,
		// Plain JSON Schema; pi validates it via its non-TypeBox branch.
		parameters: (tool.inputSchema ?? { type: "object", properties: {} }) as TSchema,
		execute: async (_toolCallId, params, signal): Promise<AgentToolResult<unknown>> => {
			const result = await client.callTool(
				{ name: tool.name, arguments: (params ?? {}) as Record<string, unknown> },
				undefined,
				{ timeout: CALL_TIMEOUT_MS, signal },
			);
			const content = mapContent(result.content);
			if (result.isError) {
				const text = content
					.map((block) => (block.type === "text" ? block.text : "<image>"))
					.join("\n")
					.trim();
				throw new Error(text || `MCP tool ${tool.name} failed`);
			}
			return {
				content,
				details: result.structuredContent ?? undefined,
			};
		},
	};
}

export class McpHost {
	private servers: ConnectedServer[] = [];

	/**
	 * Connect to each configured server. Per-server failures are collected and
	 * reported, not fatal: one broken server must not take the seat down.
	 */
	async connect(configs: McpServerConfig[]): Promise<Record<string, string>> {
		const errors: Record<string, string> = {};
		for (const config of configs) {
			if (!NAME_PATTERN.test(config.name)) {
				errors[config.name] = `Invalid server name "${config.name}" (allowed: [a-zA-Z0-9_-]+)`;
				continue;
			}
			if (this.servers.some((server) => server.name === config.name)) {
				errors[config.name] = "Duplicate server name";
				continue;
			}
			try {
				const client = new Client({ name: "shodh-seat-harness", version: "0.1.0" });
				const transport = new StdioClientTransport({
					command: config.command,
					args: config.args ?? [],
					env: { ...getDefaultEnvironment(), ...(config.env ?? {}) },
					cwd: config.cwd,
					stderr: "ignore",
				});
				await client.connect(transport);
				const listed = await client.listTools();
				const tools = listed.tools.map((tool) => toAgentTool(config.name, client, tool));
				this.servers.push({ name: config.name, client, tools });
			} catch (error) {
				errors[config.name] = error instanceof Error ? error.message : String(error);
			}
		}
		return errors;
	}

	/** All bridged tools across connected servers. */
	getTools(): AgentTool<any>[] {
		return this.servers.flatMap((server) => server.tools);
	}

	listServers(): { name: string; tool_count: number }[] {
		return this.servers.map((server) => ({ name: server.name, tool_count: server.tools.length }));
	}

	async close(): Promise<void> {
		await Promise.allSettled(this.servers.map((server) => server.client.close()));
		this.servers = [];
	}
}
