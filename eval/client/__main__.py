"""CLI: connect to a K8sGPT MCP server, initialize, and list tools.

Run with:  uv run python -m eval.client
Override URL with KUBELM_MCP_URL.
"""

from __future__ import annotations

import json
import sys

from eval import K8SGPT_VERSION
from eval.client.mcp import MCPClient


def main() -> int:
    client = MCPClient()
    print(f"-> {client.url}  initialize  (pinned K8sGPT {K8SGPT_VERSION})")
    client.initialize()
    print(
        json.dumps(
            {
                "serverInfo": client.server_info,
                "capabilities": client.server_capabilities,
            },
            indent=2,
        )
    )

    tools = client.list_tools()
    print(f"-> {client.url}  tools/list  ({len(tools)} tools)")
    for t in tools.values():
        required = t.input_schema.get("required", [])
        print(f"  {t.name:30s}  required={required}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
