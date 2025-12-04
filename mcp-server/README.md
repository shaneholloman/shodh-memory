# Shodh-Memory MCP Server

Give Claude Code persistent memory using Shodh-Memory. Built with Bun.

## Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a memory (observation, decision, learning, etc.) |
| `recall` | Semantic search to find relevant memories |
| `list_memories` | List all stored memories |
| `forget` | Delete a specific memory |
| `memory_stats` | Get memory statistics |

## Setup

### 1. Start shodh-memory server

```bash
./shodh-memory-server
```

### 2. Install dependencies

```bash
cd mcp-server
bun install
```

### 3. Add to Claude Code config

Edit `~/.claude.json`:

```json
{
  "mcpServers": {
    "shodh-memory": {
      "command": "bun",
      "args": ["run", "/path/to/shodh-memory/mcp-server/index.ts"],
      "env": {
        "SHODH_API_URL": "http://127.0.0.1:3030",
        "SHODH_USER_ID": "claude-code"
      }
    }
  }
}
```

## Usage

Once configured, Claude can:

```
"Remember that the user prefers Rust over Python"
"Recall what I know about user preferences"
"List all my memories"
"Show memory stats"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SHODH_API_URL` | `http://127.0.0.1:3030` | Shodh-Memory server URL |
| `SHODH_API_KEY` | `shodh-dev-key...` | API key |
| `SHODH_USER_ID` | `claude-code` | User ID for memory isolation |
