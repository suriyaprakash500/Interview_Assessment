"""MCP server exposing SQLite database through read-only tools."""

import os
import re
import json
import sqlite3
from mcp.server.fastmcp import FastMCP

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "company.db")

mcp = FastMCP("CompanyDB")

# Dangerous SQL keywords (case-insensitive)
DANGEROUS_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE)\b", re.IGNORECASE
)


def _get_connection():
    """Create a read-only database connection."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def list_tables() -> str:
    """List all tables in the database."""
    conn = _get_connection()
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cur.fetchall()]
        return json.dumps(tables)
    finally:
        conn.close()


@mcp.tool()
def describe_schema(table_name: str) -> str:
    """Describe the schema of a table, including column names and types."""
    conn = _get_connection()
    try:
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        columns = [
            {"name": row["name"], "type": row["type"], "primary_key": bool(row["pk"])}
            for row in cur.fetchall()
        ]
        if not columns:
            return json.dumps({"error": f"Table '{table_name}' not found"})

        # Also get foreign key info
        fk_cur = conn.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = [
            {"from": row["from"], "to_table": row["table"], "to_column": row["to"]}
            for row in fk_cur.fetchall()
        ]

        return json.dumps({"table": table_name, "columns": columns, "foreign_keys": foreign_keys})
    finally:
        conn.close()


@mcp.tool()
def run_query(sql: str) -> str:
    """Execute a read-only SQL query and return results as JSON."""
    stripped = sql.strip()
    upper = stripped.upper()

    # Defense layer 1: must start with SELECT or WITH
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return json.dumps({"error": "Only SELECT and WITH queries are allowed."})

    # Defense layer 2: reject dangerous keywords
    if DANGEROUS_PATTERN.search(stripped):
        return json.dumps({"error": "Query contains forbidden keywords. Only read-only queries are allowed."})

    # Auto-add LIMIT 100 if not present
    if "LIMIT" not in upper:
        stripped = stripped.rstrip(";")
        stripped += " LIMIT 100"

    conn = _get_connection()
    try:
        cur = conn.execute(stripped)
        rows = [dict(row) for row in cur.fetchall()]
        return json.dumps({"rows": rows, "count": len(rows)})
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run()
