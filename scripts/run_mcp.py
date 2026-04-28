#!/usr/bin/env python
"""Launch the CHF titration MCP server (stdio transport for Claude Code / Claude Desktop).

Usage:
    python scripts/run_mcp.py
"""
from chf_titration.mcp_server import mcp

if __name__ == "__main__":
    mcp.run()
