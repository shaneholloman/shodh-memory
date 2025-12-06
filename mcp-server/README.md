# Shodh-Memory MCP Server

Persistent AI memory with semantic search. Store observations, decisions, learnings, and recall them across sessions.

## Features

- **Semantic Search**: Find memories by meaning, not just keywords
- **Memory Types**: Categorize as Observation, Decision, Learning, Error, Pattern, etc.
- **Persistent**: Memories survive across sessions and restarts
- **Fast**: Sub-millisecond retrieval with vector indexing

## Installation

### 1. Start the shodh-memory backend

```bash
# Download and run the server
cargo install shodh-memory
shodh-memory-server
```

Or with Docker:
```bash
docker run -p 3030:3030 shodh/memory
```

### 2. Configure your MCP client

**For Claude Desktop** (`~/.claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "shodh-memory": {
      "command": "npx",
      "args": ["@shodh/memory-mcp"],
      "env": {
        "SHODH_API_URL": "http://127.0.0.1:3030"
      }
    }
  }
}
```

**For Cursor/other MCP clients**: Similar configuration with the npx command.

## Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a memory with optional type and tags |
| `recall` | Semantic search to find relevant memories |
| `list_memories` | List all stored memories |
| `forget` | Delete a specific memory by ID |
| `memory_stats` | Get statistics about stored memories |

## Usage Examples

```
"Remember that the user prefers Rust over Python for systems programming"
"Recall what I know about user's programming preferences"
"List my recent memories"
"Show memory stats"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SHODH_API_URL` | `http://127.0.0.1:3030` | Backend server URL |
| `SHODH_API_KEY` | (dev key) | API key for authentication |
| `SHODH_USER_ID` | `default` | User ID for memory isolation |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   AI Client     │────▶│   MCP Server    │────▶│ Shodh Backend   │
│ (Claude, etc.)  │     │  (this package) │     │  (Rust server)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              stdio              REST API
```

## License

Apache-2.0
