# MCP SQL Agent

A natural-language database assistant built with the **Model Context Protocol (MCP)**.  
An LLM agent answers questions about a company database by discovering schema and generating SQL dynamically — all through MCP tools, never touching the database directly.

---

## Architecture

```
┌────────────┐
│    User    │
└─────┬──────┘
      │  natural language question
      ▼
┌────────────┐
│   Agent    │  (Groq / Llama 3.1)
│            │  - Discovers schema at runtime
│            │  - Generates SQL from questions
│            │  - Retries on errors
└─────┬──────┘
      │  MCP tool calls (stdio)
      ▼
┌────────────┐
│ MCP Server │  - list_tables()
│            │  - describe_schema()
│            │  - run_query()
│            │  - Enforces read-only access
│            │  - Applies row limits
└─────┬──────┘
      │  SQL
      ▼
┌────────────┐
│   SQLite   │  company.db
│            │  employees / projects / issues
└────────────┘
```

---

## MCP as a Connector Layer Between LLM and Database

MCP (Model Context Protocol) acts as an abstraction layer between the LLM and the database. Instead of giving the LLM a raw connection string or letting it execute arbitrary SQL directly, MCP exposes a narrow, well-defined set of tools — `list_tables`, `describe_schema`, and `run_query`. This is fundamentally better than direct database access for several reasons:

1. **Security boundary**: The LLM never sees credentials. Connection strings, file paths, and authentication details exist only inside the MCP server process. Even if the LLM is prompt-injected, it cannot leak what it doesn't have.

2. **Centralized enforcement**: Read-only policies, row limits, SQL validation, and auditing are enforced in one place (the MCP server), not scattered across prompts or agent code. Adding a new guardrail means changing one file.

3. **Protocol-level isolation**: The agent communicates with the MCP server over **stdio transport** using JSON-RPC. The agent spawns the server as a subprocess — it only knows how to call tools, not where or how data is stored. This is analogous to a REST API: the client calls endpoints, not raw database queries.

4. **Composability**: The same MCP server can serve multiple agents or UIs without duplicating security logic. Swapping SQLite for PostgreSQL requires changes only in the MCP server — the agent code doesn't change at all.

Without MCP, you would need to embed connection strings in agent code, trust the LLM's prompt compliance for safety, and re-implement guardrails for every new agent — all of which are fragile and audit-unfriendly.

---

## Why Credentials Stay Out of Agent Code

The database path (`company.db`) exists **only** inside `mcp_server/server.py`. The agent code (`agent/agent.py`, `agent/llm.py`) never imports `sqlite3`, never constructs a connection string, and never references a file path. This separation means:

- The agent prompt cannot leak credentials
- The LLM cannot be tricked into revealing connection details
- Database access is gated through a narrow, auditable API

---

## Read-Only Enforcement (Defense in Depth)

Four layers prevent write operations — if any one layer is bypassed, the others still block mutations:

1. **Prompt-level**: The system prompt instructs the LLM to generate only `SELECT` queries.
2. **MCP-tool-level validation**:
   - SQL must start with `SELECT` or `WITH` (allows CTEs, rejects everything else)
   - Regex rejects dangerous keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`
3. **DB-permission-level**: The SQLite connection is opened in **read-only mode** (`?mode=ro`), so even if malicious SQL bypasses both checks above, the database engine itself will reject writes.
4. **Credential isolation**: Database access exists only inside the MCP server — the agent has no way to open a direct connection, so it cannot circumvent any of the above.

---

## Schema Discovery and Introspection Trade-off

The agent does **not** hardcode table names or column names. On every question:

1. Calls `list_tables()` → discovers available tables
2. Calls `describe_schema(table)` for each table → gets columns, types, primary keys, foreign keys
3. Sends the full schema context to the LLM along with the question
4. The LLM generates SQL using the actual column names

This means the agent adapts automatically if the database schema changes.

**Design trade-off — full schema vs. on-demand introspection:**

This implementation sends the full schema (all tables) to the LLM on every question. For a small database (3 tables), this is the right choice: it costs ~200 extra tokens but gives the LLM complete context to plan JOINs correctly in a single step. For a large database (100+ tables), you would switch to on-demand introspection — only describing tables the LLM selects after seeing the table list — to avoid blowing the context window. The current approach favors accuracy and simplicity over token cost, which is appropriate for this scale.

---

## Error Recovery

If a generated SQL query fails (e.g., wrong column name):

```
Attempt 1:  SELECT employee_name FROM employees WHERE department = 'AI'
  → Error: no such column: employee_name

Agent re-inspects schema...

Attempt 2:  SELECT name FROM employees WHERE department = 'AI'
  → Success ✓
```

The agent:
1. Reads the error message
2. Re-discovers the schema via MCP
3. Sends both the error and fresh schema to the LLM
4. The LLM generates a corrected query
5. Maximum 2 retries (3 total attempts)

---

## Query Guardrails

- All queries are automatically capped at **100 rows** (`LIMIT 100` is appended if missing)
- This prevents runaway queries and controls cost

---

## Project Structure

```
Assignment-2/
├── database/
│   └── setup_db.py          # Creates and populates company.db
├── mcp_server/
│   └── server.py            # MCP server with 3 tools
├── agent/
│   ├── llm.py               # Groq API wrapper
│   └── agent.py             # Plan→Act→Observe agent
├── demo.py                  # Runs 4 demo examples
├── requirements.txt
├── .env.example
└── README.md
```

---

## How to Run

> **Requires Python 3.10+** (the `mcp` package does not support older versions).

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### 3. Create the database

```bash
python database/setup_db.py
```

### 4. Run the demo

```bash
python demo.py
```

The demo runs 4 examples:

| # | Example | What it demonstrates |
|---|---------|---------------------|
| 1 | "Fetch employee details where department = AI" | Basic single-table query |
| 2 | "Which AI-team members have open issues on Project X?" | Multi-table JOIN across all 3 tables |
| 3 | "Show employee_name for AI employees" | Error recovery (bad column → retry) |
| 4 | "Show employees" | Clarification workflow |

Each example prints all intermediate agent steps: `[PLAN]`, `[ACTION]`, `[OBSERVATION]`, `[ANSWER]`.

---

## Technology Stack

- **LLM**: Groq API with `llama-3.1-8b-instant`
- **MCP**: Official Python SDK (`mcp` package) with stdio transport
- **Database**: SQLite
- **Language**: Python 3.10+
