"""Agent that answers natural-language questions using MCP tools and Groq LLM."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent.llm import chat

MAX_RETRIES = 2

SYSTEM_PROMPT = """You are a database assistant. You answer questions by generating SQL queries.

You will be given:
- A database schema (tables, columns, types, foreign keys)
- A user question

You must respond with EXACTLY one JSON object (no extra text):

If you can generate SQL:
{"action": "query", "sql": "YOUR SQL HERE"}

If the question is ambiguous and you need clarification (e.g. the user says "show employees" without specifying a filter):
{"action": "clarify", "message": "Your clarification question here"}

Rules:
- Generate only read-only SELECT queries
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE
- Use JOINs when the question involves multiple tables
- Always use exact column names from the schema provided
- Output ONLY the JSON object, nothing else
"""

RETRY_PROMPT = """The previous SQL query failed with this error:
{error}

Here is the database schema again:
{schema}

Please generate a corrected SQL query. Respond with EXACTLY one JSON object:
{{"action": "query", "sql": "YOUR CORRECTED SQL HERE"}}
"""


def _print_step(label: str, content: str):
    """Print a labeled agent step."""
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"{'-'*60}")
    print(content)
    print(f"{'='*60}")


class MCPAgent:
    def __init__(self, step_callback=None):
        self._session = None
        self._read = None
        self._write = None
        self.steps = []  # Collected steps for UI display
        self._step_callback = step_callback  # Optional callback for real-time updates

    async def connect(self):
        """Connect to the MCP server via stdio."""
        server_script = os.path.join(
            os.path.dirname(__file__), "..", "mcp_server", "server.py"
        )
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script],
        )
        # Store the context managers so we can clean up later
        self._stdio_cm = stdio_client(server_params)
        self._read, self._write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(self._read, self._write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        print("Connected to MCP server.")

    async def close(self):
        """Close the MCP connection."""
        if self._session_cm:
            await self._session_cm.__aexit__(None, None, None)
        if self._stdio_cm:
            await self._stdio_cm.__aexit__(None, None, None)

    async def _call_tool(self, name: str, arguments: dict = None) -> str:
        """Call an MCP tool and return the text result."""
        result = await self._session.call_tool(name, arguments or {})
        # MCP returns content blocks; extract text
        return result.content[0].text

    def _log(self, label: str, content: str):
        """Log a step: print to console and collect for UI."""
        _print_step(label, content)
        self.steps.append({"label": label, "content": content})
        if self._step_callback:
            self._step_callback(label, content)

    async def _discover_schema(self) -> str:
        """Discover database schema via MCP tools. Returns formatted schema string."""
        self._log("ACTION", "Calling list_tables()")
        tables_json = await self._call_tool("list_tables")
        tables = json.loads(tables_json)
        self._log("OBSERVATION", f"Tables found: {tables}")

        schema_parts = []
        for table in tables:
            self._log("ACTION", f"Calling describe_schema('{table}')")
            schema_json = await self._call_tool("describe_schema", {"table_name": table})
            schema = json.loads(schema_json)
            self._log("OBSERVATION", f"Schema for {table}: {json.dumps(schema, indent=2)}")
            schema_parts.append(json.dumps(schema, indent=2))

        return "\n\n".join(schema_parts)

    async def ask(self, question: str, followup: str = None, initial_sql: str = None) -> str:
        """Answer a natural-language question using MCP tools.

        Args:
            question: The user's question.
            followup: Optional follow-up answer to a clarification question.
            initial_sql: If provided, skip LLM and try this SQL first (for error recovery demo).

        Returns:
            The final natural-language answer.
        """
        self.steps = []  # Reset steps for this question
        print(f"\n{'#'*60}")
        print(f"QUESTION: {question}")
        if followup:
            print(f"FOLLOW-UP: {followup}")
        print(f"{'#'*60}")

        # Step 1-2: Discover schema
        self._log("PLAN", "Step 1: Discover database schema via MCP tools\n"
                           "Step 2: Generate SQL from the question\n"
                           "Step 3: Execute SQL and summarize results")

        schema_text = await self._discover_schema()

        # Build the prompt for the LLM
        user_message = f"Database schema:\n{schema_text}\n\nQuestion: {question}"
        if followup:
            user_message += f"\nAdditional context: {followup}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # If initial_sql is provided, skip LLM and try it directly (for error recovery demo)
        if initial_sql:
            sql = initial_sql
            self._log("ACTION", f"Using provided SQL: {sql}")
        else:
            # Step 3: Ask LLM to generate SQL (or clarify)
            self._log("ACTION", "Sending question and schema to LLM")
            llm_response = chat(messages)
            self._log("OBSERVATION", f"LLM response: {llm_response}")

            try:
                action = json.loads(llm_response)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                start = llm_response.find("{")
                end = llm_response.rfind("}") + 1
                if start != -1 and end > start:
                    action = json.loads(llm_response[start:end])
                else:
                    return f"LLM returned invalid response: {llm_response}"

            # Handle clarification
            if action.get("action") == "clarify":
                self._log("ANSWER", f"Clarification needed: {action['message']}")
                return f"CLARIFICATION: {action['message']}"

            sql = action.get("sql", "")

        # Step 4-5: Execute SQL with retry logic
        for attempt in range(1, MAX_RETRIES + 2):  # attempts 1, 2, 3
            self._log("ACTION", f"Attempt {attempt}: Calling run_query('{sql}')")
            result_json = await self._call_tool("run_query", {"sql": sql})
            result = json.loads(result_json)
            self._log("OBSERVATION", f"Query result: {json.dumps(result, indent=2)}")

            if "error" not in result:
                break  # success

            if attempt > MAX_RETRIES:
                return f"Query failed after {MAX_RETRIES + 1} attempts. Last error: {result['error']}"

            # Error recovery: re-discover schema and ask LLM to fix
            self._log("PLAN", f"Query failed: {result['error']}\n"
                               f"Re-inspecting schema and retrying (attempt {attempt + 1})...")
            schema_text = await self._discover_schema()
            retry_msg = RETRY_PROMPT.format(error=result["error"], schema=schema_text)
            messages.append({"role": "assistant", "content": json.dumps({"action": "query", "sql": sql})})
            messages.append({"role": "user", "content": retry_msg})

            llm_response = chat(messages)
            self._log("OBSERVATION", f"LLM retry response: {llm_response}")

            try:
                action = json.loads(llm_response)
            except json.JSONDecodeError:
                start = llm_response.find("{")
                end = llm_response.rfind("}") + 1
                if start != -1 and end > start:
                    action = json.loads(llm_response[start:end])
                else:
                    return f"LLM returned invalid response on retry: {llm_response}"

            sql = action.get("sql", "")

        # Step 6: Summarize results in natural language
        summary_messages = [
            {"role": "system", "content": "Summarize the following database query results in clear, natural language. Be concise."},
            {"role": "user", "content": f"Question: {question}\nSQL: {sql}\nResults: {json.dumps(result)}"},
        ]
        answer = chat(summary_messages)
        self._log("ANSWER", answer)
        return answer
