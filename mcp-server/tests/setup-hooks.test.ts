import { describe, expect, it } from "vitest";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
// Requiring the CLI module must NOT execute its main() (guarded by require.main).
const setupHooks = require("../scripts/setup-hooks.cjs") as {
  makeCommand: (event: string) => string;
  buildHookConfig: () => Record<string, Array<{ matcher?: unknown; hooks: Array<{ type: string; command: string }> }>>;
  mergeHooks: (settings: Record<string, unknown>, hooks: Record<string, unknown>) => { hooks: Record<string, unknown> };
  removeHooks: (settings: Record<string, unknown>) => { hooks?: Record<string, unknown> };
  HOOK_SCRIPT: string;
};

const { makeCommand, buildHookConfig, mergeHooks, removeHooks, HOOK_SCRIPT } = setupHooks;

describe("makeCommand — T17 path quoting", () => {
  it("quotes the script path so paths containing spaces survive shell parsing", () => {
    const cmd = makeCommand("SessionStart");
    const scriptPath = HOOK_SCRIPT.replace(/\\/g, "/");
    expect(cmd).toBe(`bun run "${scriptPath}" SessionStart`);
  });

  it("produces a parseable command even when the home directory contains spaces", () => {
    const cmd = makeCommand("Stop");
    // The path segment must be wrapped in double quotes: exactly one quoted
    // region between the runner prefix and the event argument.
    const match = /^bun run "([^"]+)" Stop$/.exec(cmd);
    expect(match).not.toBeNull();
    expect(match![1].endsWith("memory-hook.ts")).toBe(true);
  });
});

describe("buildHookConfig — T17 matcher shape", () => {
  const config = buildHookConfig();

  it("uses Claude Code's string matcher for tool events", () => {
    expect(config.PreToolUse[0].matcher).toBe("Edit|Write|Bash");
    expect(config.PostToolUse[0].matcher).toBe("Edit|Write|Bash|TodoWrite|Read|Task");
  });

  it("omits matcher entirely for non-tool events", () => {
    for (const event of ["SessionStart", "UserPromptSubmit", "Stop", "SubagentStop"]) {
      expect(Object.prototype.hasOwnProperty.call(config[event][0], "matcher")).toBe(false);
    }
  });

  it("never emits the legacy object matcher shape", () => {
    for (const entries of Object.values(config)) {
      for (const entry of entries) {
        if (entry.matcher !== undefined) {
          expect(typeof entry.matcher).toBe("string");
        }
      }
    }
  });
});

describe("mergeHooks / removeHooks", () => {
  it("merge is idempotent (same command not appended twice)", () => {
    const config = buildHookConfig();
    const once = mergeHooks({}, config);
    const twice = mergeHooks(once, config);
    expect(twice).toEqual(once);
  });

  it("preserves unrelated hooks", () => {
    const settings = {
      hooks: {
        Stop: [{ hooks: [{ type: "command", command: "echo other-tool" }] }],
      },
    };
    const merged = mergeHooks(settings, buildHookConfig());
    const stopEntries = merged.hooks.Stop as Array<{ hooks: Array<{ command: string }> }>;
    expect(stopEntries.some((e) => e.hooks[0].command === "echo other-tool")).toBe(true);
    expect(stopEntries.some((e) => e.hooks[0].command.includes("memory-hook.ts"))).toBe(true);
  });

  it("removeHooks strips legacy broken entries (unquoted path, object matcher)", () => {
    const legacy = {
      hooks: {
        PostToolUse: [
          {
            matcher: { tool_name: ["Edit", "Write"] },
            hooks: [{ type: "command", command: "bun run C:/Users/John Smith/.claude/hooks/shodh-memory/memory-hook.ts PostToolUse" }],
          },
          { matcher: "Edit", hooks: [{ type: "command", command: "echo keep-me" }] },
        ],
      },
    };
    const cleaned = removeHooks(legacy);
    const entries = cleaned.hooks!.PostToolUse as Array<{ hooks: Array<{ command: string }> }>;
    expect(entries).toHaveLength(1);
    expect(entries[0].hooks[0].command).toBe("echo keep-me");
  });

  it("upgrade path: removeHooks + mergeHooks replaces a broken install with the fixed shape", () => {
    const legacy = {
      hooks: {
        PreToolUse: [
          {
            matcher: { tool_name: ["Edit", "Write", "Bash"] },
            hooks: [{ type: "command", command: "bun run C:/Users/John Smith/.claude/hooks/shodh-memory/memory-hook.ts PreToolUse" }],
          },
        ],
      },
    };
    const repaired = mergeHooks(removeHooks(legacy), buildHookConfig());
    const entries = repaired.hooks.PreToolUse as Array<{ matcher?: unknown; hooks: Array<{ command: string }> }>;
    expect(entries).toHaveLength(1);
    expect(entries[0].matcher).toBe("Edit|Write|Bash");
    expect(entries[0].hooks[0].command).toMatch(/^bun run "/);
  });
});
