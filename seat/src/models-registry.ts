/**
 * Model registry: pi's full built-in provider catalog plus local
 * OpenAI-compatible endpoints (Ollama, LM Studio) as first-class providers.
 *
 * pi ships no local-model provider (packages/ai/src/providers/ has none), but
 * every `Model` carries its own `baseUrl` (packages/ai/src/types.ts) and the
 * openai-completions API implementation uses it directly
 * (packages/ai/src/api/openai-completions.ts → `baseURL: model.baseUrl`), so a
 * local endpoint is just a custom provider built with `createProvider` +
 * `openAICompletionsApi()` and dynamic model discovery via GET {baseUrl}/models.
 *
 * Provider API keys resolve from environment variables inside this process
 * (pi's env auth). They are never sent to, or readable by, any client.
 */

import {
	type Api,
	createProvider,
	type Model,
	type MutableModels,
	type Provider,
	type RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";
import type { SeatConfig } from "./config.js";

export interface ModelInfo {
	provider: string;
	id: string;
	name: string;
	context_window: number;
	max_tokens: number;
	reasoning: boolean;
	local: boolean;
}

const LOCAL_PROVIDER_IDS = ["ollama", "lmstudio"] as const;

interface LocalProviderOptions {
	id: (typeof LOCAL_PROVIDER_IDS)[number];
	name: string;
	baseUrl: string;
	contextWindow: number;
	maxTokens: number;
}

/** Response shape of the OpenAI-compatible model listing endpoint (GET /v1/models). */
interface OpenAiModelList {
	data?: { id?: unknown }[];
}

function localModel(options: LocalProviderOptions, id: string): Model<"openai-completions"> {
	return {
		id,
		name: id,
		api: "openai-completions",
		provider: options.id,
		baseUrl: options.baseUrl,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: options.contextWindow,
		maxTokens: options.maxTokens,
	};
}

async function fetchLocalModels(
	options: LocalProviderOptions,
	context: RefreshModelsContext,
): Promise<readonly Model<"openai-completions">[]> {
	const response = await fetch(`${options.baseUrl}/models`, {
		signal: AbortSignal.any([context.signal, AbortSignal.timeout(5000)]),
	});
	if (!response.ok) {
		throw new Error(`${options.name} model listing failed: HTTP ${response.status}`);
	}
	const payload = (await response.json()) as OpenAiModelList;
	const ids = (payload.data ?? [])
		.map((entry) => entry.id)
		.filter((id): id is string => typeof id === "string" && id.length > 0);
	return ids.map((id) => localModel(options, id));
}

function localProvider(options: LocalProviderOptions): Provider<"openai-completions"> {
	return createProvider<"openai-completions">({
		id: options.id,
		name: options.name,
		baseUrl: options.baseUrl,
		auth: {
			// Keyless local endpoint: resolve always succeeds so the provider is
			// "configured"; the placeholder key satisfies clients that require
			// a bearer token (Ollama and LM Studio both ignore it).
			apiKey: {
				name: `${options.name} endpoint`,
				resolve: async () => ({ auth: { apiKey: "local" }, source: "local endpoint (keyless)" }),
			},
		},
		models: [],
		fetchModels: (context) => fetchLocalModels(options, context),
		api: openAICompletionsApi(),
	});
}

export class ModelRegistry {
	readonly models: MutableModels;

	constructor(config: SeatConfig) {
		this.models = builtinModels();
		this.models.setProvider(
			localProvider({
				id: "ollama",
				name: "Ollama",
				baseUrl: config.ollamaBaseUrl,
				contextWindow: config.localContextWindow,
				maxTokens: config.localMaxTokens,
			}),
		);
		this.models.setProvider(
			localProvider({
				id: "lmstudio",
				name: "LM Studio",
				baseUrl: config.lmStudioBaseUrl,
				contextWindow: config.localContextWindow,
				maxTokens: config.localMaxTokens,
			}),
		);
	}

	/**
	 * Refresh local-endpoint model listings. Errors are returned, not thrown:
	 * an offline Ollama must not take the seat down.
	 */
	async refreshLocal(): Promise<Record<string, string>> {
		const result = await this.models.refresh({ providers: [...LOCAL_PROVIDER_IDS] });
		const errors: Record<string, string> = {};
		for (const [provider, error] of result.errors) {
			errors[provider] = error.message;
		}
		return errors;
	}

	/** Models whose providers have working auth (env keys present, or local). */
	async listAvailable(): Promise<ModelInfo[]> {
		const available = await this.models.getAvailable();
		return available.map((model) => ({
			provider: model.provider,
			id: model.id,
			name: model.name,
			context_window: model.contextWindow,
			max_tokens: model.maxTokens,
			reasoning: model.reasoning,
			local: (LOCAL_PROVIDER_IDS as readonly string[]).includes(model.provider),
		}));
	}

	resolve(provider: string, id: string): Model<Api> | undefined {
		return this.models.getModel(provider, id);
	}
}
