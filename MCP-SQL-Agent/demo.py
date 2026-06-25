"""Demo script showing the MCP SQL Agent in action."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from database.setup_db import setup_database
from agent.agent import MCPAgent


async def run_demo():
    # Step 0: Ensure database exists
    print("Setting up database...")
    setup_database()
    print()

    agent = MCPAgent()
    await agent.connect()

    try:
        # ── Example 1: Basic query ─────────────────────────────────
        print("\n" + "=" * 60)
        print("  EXAMPLE 1: Basic department query")
        print("=" * 60)
        await agent.ask("Fetch employee details where department = AI")

        # ── Example 2: Multi-table JOIN ────────────────────────────
        print("\n" + "=" * 60)
        print("  EXAMPLE 2: Multi-table JOIN query")
        print("=" * 60)
        await agent.ask("Which AI-team members have open issues on Project X?")

        # ── Example 3: Error recovery ──────────────────────────────
        print("\n" + "=" * 60)
        print("  EXAMPLE 3: Error recovery (intentional bad column name)")
        print("=" * 60)
        await agent.ask(
            "Fetch employee details where department = AI",
            initial_sql="SELECT employee_name, department, email FROM employees WHERE department = 'AI'"
        )

        # ── Example 4: Clarification flow ──────────────────────────
        print("\n" + "=" * 60)
        print("  EXAMPLE 4: Clarification workflow")
        print("=" * 60)
        result = await agent.ask("Show employees")

        if result.startswith("CLARIFICATION:"):
            print(f"\nAgent asked for clarification. Providing follow-up: 'AI'")
            await agent.ask("Show employees", followup="The department is AI")

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(run_demo())
