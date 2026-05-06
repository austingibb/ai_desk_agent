"""Minimal MCP JSON-RPC client for Brave Search MCP server over SSE/HTTP transport."""

import json
import requests

MCP_URL = "http://192.168.0.4:8089/mcp"
JSONRPC = "2.0"


class MCPClient:
    def __init__(self, url: str = MCP_URL, timeout: int = 30):
        self.url = url
        self.timeout = timeout
        self._request_id = 0
        self._initialized = False
        self._tools = []

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _rpc(self, method: str, params: dict = None) -> dict:
        body = {
            "jsonrpc": JSONRPC,
            "method": method,
            "params": params or {},
            "id": self._next_id(),
        }
        resp = requests.post(
            self.url,
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()

        raw = resp.text
        if raw.startswith("event:"):
            for line in raw.split("\n"):
                if line.startswith("data:"):
                    return json.loads(line[5:].strip())
            raise ValueError(f"Could not parse SSE response: {raw[:200]}")
        return resp.json()

    def initialize(self):
        result = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ai-friend", "version": "1.0"},
        })
        if "error" in result:
            raise RuntimeError(f"MCP initialize failed: {result['error']}")
        self._initialized = True

        tools = self._rpc("tools/list")
        if "error" in tools:
            raise RuntimeError(f"MCP tools/list failed: {tools['error']}")
        self._tools = tools.get("result", {}).get("tools", [])
        return self._tools

    @property
    def tools(self) -> list:
        return self._tools

    def get_tool_definitions(self) -> list:
        """Return MCP tools converted to OpenAI function-calling format."""
        defs = []
        for tool in self._tools:
            params = tool.get("inputSchema", {})
            if "additionalProperties" not in params:
                params["additionalProperties"] = False
            defs.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": params,
                },
            })
        return defs

    def call_tool(self, name: str, arguments: dict) -> dict:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if "error" in result:
            return {"error": str(result["error"])}

        content = result.get("result", {}).get("content", [])
        if content and content[0].get("type") == "text":
            try:
                return json.loads(content[0]["text"])
            except json.JSONDecodeError:
                return {"text": content[0]["text"]}
        return {"raw": content}
