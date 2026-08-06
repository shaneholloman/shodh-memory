import { describe, expect, it, beforeEach, afterEach } from "vitest";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
  API_KEY_FILENAME,
  apiKeyFilePath,
  loadOrCreatePersistedApiKey,
  readPersistedApiKey,
  shodhDataRoot,
} from "../api-key-store";

let tmpRoot: string;

beforeEach(() => {
  tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "shodh-key-test-"));
});

afterEach(() => {
  fs.rmSync(tmpRoot, { recursive: true, force: true });
});

describe("shodhDataRoot", () => {
  it("prefers SHODH_MEMORY_PATH when set", () => {
    const root = shodhDataRoot({ SHODH_MEMORY_PATH: "/custom/data" } as NodeJS.ProcessEnv, "linux", "/home/u");
    expect(root).toBe("/custom/data");
  });

  it("uses %APPDATA%/shodh-memory on Windows (matches Rust dirs::data_dir)", () => {
    const root = shodhDataRoot(
      { APPDATA: "C:\\Users\\John Smith\\AppData\\Roaming" } as NodeJS.ProcessEnv,
      "win32",
      "C:\\Users\\John Smith",
    );
    expect(root).toBe(path.join("C:\\Users\\John Smith\\AppData\\Roaming", "shodh-memory"));
  });

  it("falls back to <home>/AppData/Roaming when APPDATA is unset on Windows", () => {
    const root = shodhDataRoot({} as NodeJS.ProcessEnv, "win32", "C:\\Users\\John Smith");
    expect(root).toBe(path.join("C:\\Users\\John Smith", "AppData", "Roaming", "shodh-memory"));
  });

  it("uses Application Support on macOS", () => {
    const root = shodhDataRoot({} as NodeJS.ProcessEnv, "darwin", "/Users/u");
    expect(root).toBe(path.join("/Users/u", "Library", "Application Support", "shodh-memory"));
  });

  it("respects XDG_DATA_HOME on Linux, defaulting to ~/.local/share", () => {
    expect(shodhDataRoot({ XDG_DATA_HOME: "/xdg" } as NodeJS.ProcessEnv, "linux", "/home/u")).toBe(
      path.join("/xdg", "shodh-memory"),
    );
    expect(shodhDataRoot({} as NodeJS.ProcessEnv, "linux", "/home/u")).toBe(
      path.join("/home/u", ".local", "share", "shodh-memory"),
    );
  });
});

describe("loadOrCreatePersistedApiKey", () => {
  it("generates and persists a key when none exists", () => {
    const result = loadOrCreatePersistedApiKey(tmpRoot, () => "generated-key-1");
    expect(result.created).toBe(true);
    expect(result.key).toBe("generated-key-1");
    expect(result.file).toBe(path.join(tmpRoot, API_KEY_FILENAME));
    expect(fs.readFileSync(result.file, "utf-8").trim()).toBe("generated-key-1");
  });

  it("returns the SAME key on subsequent calls instead of regenerating (the T15 race fix)", () => {
    const first = loadOrCreatePersistedApiKey(tmpRoot, () => "first-shim-key");
    // A second shim with a different generator must get the first shim's key.
    const second = loadOrCreatePersistedApiKey(tmpRoot, () => "second-shim-key");
    expect(first.key).toBe("first-shim-key");
    expect(second.key).toBe("first-shim-key");
    expect(second.created).toBe(false);
  });

  it("never invokes the generator when a key is already persisted", () => {
    fs.mkdirSync(tmpRoot, { recursive: true });
    fs.writeFileSync(path.join(tmpRoot, API_KEY_FILENAME), "existing-key\n");
    const result = loadOrCreatePersistedApiKey(tmpRoot, () => {
      throw new Error("generator must not be called when a key exists");
    });
    expect(result.key).toBe("existing-key");
    expect(result.created).toBe(false);
  });

  it("creates the data root directory when missing", () => {
    const nested = path.join(tmpRoot, "deep", "nested", "root");
    const result = loadOrCreatePersistedApiKey(nested, () => "k");
    expect(result.created).toBe(true);
    expect(readPersistedApiKey(apiKeyFilePath(nested))).toBe("k");
  });

  it("heals an empty key file left by a crashed writer", () => {
    fs.mkdirSync(tmpRoot, { recursive: true });
    fs.writeFileSync(path.join(tmpRoot, API_KEY_FILENAME), "");
    const result = loadOrCreatePersistedApiKey(tmpRoot, () => "healed-key");
    expect(result.key).toBe("healed-key");
    expect(readPersistedApiKey(apiKeyFilePath(tmpRoot))).toBe("healed-key");
  });

  it("restricts key file permissions on POSIX", () => {
    if (process.platform === "win32") return; // mode bits are not meaningful on Windows
    const result = loadOrCreatePersistedApiKey(tmpRoot, () => "k");
    const mode = fs.statSync(result.file).mode & 0o777;
    expect(mode).toBe(0o600);
  });
});

describe("readPersistedApiKey", () => {
  it("returns null for missing or empty files, trimmed content otherwise", () => {
    const file = path.join(tmpRoot, API_KEY_FILENAME);
    expect(readPersistedApiKey(file)).toBeNull();
    fs.writeFileSync(file, "   \n");
    expect(readPersistedApiKey(file)).toBeNull();
    fs.writeFileSync(file, "  the-key \n");
    expect(readPersistedApiKey(file)).toBe("the-key");
  });
});
