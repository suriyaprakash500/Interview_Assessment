"""Create and populate the company SQLite database."""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "company.db")


def setup_database():
    """Drop and recreate all tables with sample data."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Drop existing tables
    cur.executescript("""
        DROP TABLE IF EXISTS issues;
        DROP TABLE IF EXISTS projects;
        DROP TABLE IF EXISTS employees;
    """)

    # Create tables
    cur.executescript("""
        CREATE TABLE employees (
            employee_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            email TEXT NOT NULL
        );

        CREATE TABLE projects (
            project_id INTEGER PRIMARY KEY,
            project_name TEXT NOT NULL,
            department TEXT NOT NULL
        );

        CREATE TABLE issues (
            issue_id INTEGER PRIMARY KEY,
            project_id INTEGER NOT NULL,
            employee_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(project_id),
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
        );
    """)

    # Employees (10+, across AI / Data / Web)
    employees = [
        (1, "Alice Johnson", "AI", "alice@company.com"),
        (2, "Bob Smith", "AI", "bob@company.com"),
        (3, "Charlie Lee", "AI", "charlie@company.com"),
        (4, "Diana Patel", "AI", "diana@company.com"),
        (5, "Ethan Brown", "Data", "ethan@company.com"),
        (6, "Fiona Davis", "Data", "fiona@company.com"),
        (7, "George Wilson", "Data", "george@company.com"),
        (8, "Hannah Martinez", "Web", "hannah@company.com"),
        (9, "Ian Clark", "Web", "ian@company.com"),
        (10, "Julia Adams", "Web", "julia@company.com"),
        (11, "Kevin Chen", "AI", "kevin@company.com"),
    ]
    cur.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?)", employees
    )

    # Projects (5+, including Project X)
    projects = [
        (1, "Project X", "AI"),
        (2, "Data Pipeline", "Data"),
        (3, "Web Portal", "Web"),
        (4, "ML Platform", "AI"),
        (5, "Analytics Dashboard", "Data"),
        (6, "API Gateway", "Web"),
    ]
    cur.executemany(
        "INSERT INTO projects VALUES (?, ?, ?)", projects
    )

    # Issues (15+, mix of open/closed)
    # Ensure AI employees have open issues on Project X for JOIN demo
    issues = [
        (1, 1, 1, "Fix model accuracy regression", "open"),
        (2, 1, 2, "Add batch prediction endpoint", "open"),
        (3, 1, 3, "Update training data pipeline", "closed"),
        (4, 1, 4, "Optimize inference latency", "open"),
        (5, 2, 5, "Fix ETL job timeout", "open"),
        (6, 2, 6, "Add data validation checks", "closed"),
        (7, 2, 7, "Migrate to new warehouse", "open"),
        (8, 3, 8, "Redesign landing page", "closed"),
        (9, 3, 9, "Fix mobile responsiveness", "open"),
        (10, 3, 10, "Add dark mode support", "open"),
        (11, 4, 1, "Set up model registry", "closed"),
        (12, 4, 11, "Write evaluation benchmarks", "open"),
        (13, 5, 5, "Build KPI dashboard", "open"),
        (14, 6, 8, "Implement rate limiting", "closed"),
        (15, 6, 9, "Add API versioning", "open"),
        (16, 1, 11, "Review model architecture docs", "open"),
    ]
    cur.executemany(
        "INSERT INTO issues VALUES (?, ?, ?, ?, ?)", issues
    )

    conn.commit()
    conn.close()
    print(f"Database created at {DB_PATH}")
    print(f"  {len(employees)} employees, {len(projects)} projects, {len(issues)} issues")


if __name__ == "__main__":
    setup_database()
