"""Snapshot the K8sGPT MCP `tools/list` payload to a versioned JSON file.

The training trajectory format (FORMAT.md) embeds the tool definitions
the model saw, so the trained model learns valid tool shapes. K8sGPT
exposes these dynamically via `tools/list`; we capture them once per
K8sGPT version into `data/seed/tools/<version>.json` and reuse the
file across all trajectory conversions for that version.

Usage:
    uv run python data/seed/snapshot_tools.py

Requirements:
    - Docker running.
    - `kind`, `kubectl`, `k8sgpt` on PATH (same as the eval harness).

The script:
  1. Creates a throwaway one-node kind cluster (tools/list doesn't
     need cluster workload state, just a valid kubeconfig).
  2. Boots `k8sgpt serve --mcp --mcp-http` against that kubeconfig.
  3. Calls MCP `initialize` then `tools/list`.
  4. Writes `data/seed/tools/<k8sgpt_version>.json`.
  5. Tears down the kind cluster + k8sgpt process.

Idempotent: safe to re-run; will overwrite the file for the current
K8sGPT version.
"""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path

from eval import K8SGPT_VERSION
from eval.client import MCPClient
from eval.scenarios.cluster import k8sgpt_mcp_server, kind_create_cluster, kind_delete_cluster

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "data" / "seed" / "tools"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster_name = f"kubelm-snapshot-{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory() as tmp:
        kubeconfig = Path(tmp) / "kubeconfig"
        print(f"creating throwaway kind cluster {cluster_name}", file=sys.stderr)
        kind_create_cluster(
            cluster_name,
            kubeconfig_path=kubeconfig,
            node_image="kindest/node:v1.31.4",
        )
        try:
            with k8sgpt_mcp_server(kubeconfig) as mcp_url:
                client = MCPClient(url=mcp_url)
                client.initialize()
                tools = client.list_tools()
                tool_records = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    }
                    for t in tools.values()
                ]
        finally:
            print(f"tearing down kind cluster {cluster_name}", file=sys.stderr)
            kind_delete_cluster(cluster_name)

    out_path = OUT_DIR / f"{K8SGPT_VERSION}.json"
    out_path.write_text(json.dumps(tool_records, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(tool_records)} tools to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
