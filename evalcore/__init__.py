"""evalcore: plain-Python eval logic, testable without MCP.

The MCP server (`server.py`) is a thin wrapper over the functions here.
"""

# Load variables from a local .env file if one exists. This is best-effort:
# it must never break imports (e.g. in CI where python-dotenv may be absent).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional convenience only
    pass
