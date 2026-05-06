"""HTTP MCP client for K8sGPT's MCP server.

Streamable-HTTP transport: a single POST endpoint that accepts JSON-RPC
requests and returns either a JSON body or an SSE stream containing one
JSON-RPC message. K8sGPT 0.4.32 currently returns plain JSON and does not
issue an `Mcp-Session-Id` header; we still honor it if it appears, so the
client works against any spec-compliant streamable-HTTP server.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests

from eval import MCP_PROTOCOL_VERSION

DEFAULT_URL = "http://localhost:8089/mcp"
CLIENT_NAME = "kubelm"
CLIENT_VERSION = "0.0.0"

log = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPClient:
    url: str = field(default_factory=lambda: os.environ.get("KUBELM_MCP_URL", DEFAULT_URL))
    timeout: float = 30.0
    http: requests.Session = field(default_factory=requests.Session)
    server_info: dict[str, Any] | None = None
    server_capabilities: dict[str, Any] | None = None
    mcp_session_id: str | None = None
    tools: dict[str, Tool] = field(default_factory=dict)

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.mcp_session_id:
            h["Mcp-Session-Id"] = self.mcp_session_id
        return h

    def _decode(self, resp: requests.Response) -> dict[str, Any] | None:
        if not resp.content:
            return None
        ctype = resp.headers.get("Content-Type", "")
        if "text/event-stream" in ctype:
            for line in resp.text.splitlines():
                if line.startswith("data: "):
                    return json.loads(line[6:])
            raise RuntimeError(f"no data line in SSE response: {resp.text!r}")
        return resp.json()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        resp = self.http.post(
            self.url, json=payload, headers=self._headers(), timeout=self.timeout
        )
        resp.raise_for_status()
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self.mcp_session_id = sid
        return self._decode(resp)

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        result = self._post(payload)
        if result is None:
            raise RuntimeError(f"empty response for {method}")
        if "error" in result:
            raise RuntimeError(f"{method} failed: {result['error']}")
        return result["result"]

    def initialize(self) -> None:
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
        )
        self.server_info = result.get("serverInfo")
        self.server_capabilities = result.get("capabilities")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def list_tools(self) -> dict[str, Tool]:
        result = self._rpc("tools/list")
        self.tools = {
            t["name"]: Tool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                annotations=t.get("annotations", {}),
            )
            for t in result.get("tools", [])
        }
        return self.tools

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
