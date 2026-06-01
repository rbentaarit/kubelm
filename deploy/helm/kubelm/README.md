# kubelm Helm chart

Deploys a kubelm tool-use model behind an OpenAI-compatible endpoint
(llama.cpp `llama-server`, CPU-only) for K8sGPT to use as its LLM
backend. Pinned to K8sGPT **v0.4.32**.

```bash
# Model server only (point your own K8sGPT at it):
helm install kubelm deploy/helm/kubelm -n kubelm --create-namespace

# Turnkey loop (also deploys K8sGPT + an agent; submit a goal, kubelm investigates):
helm install kubelm deploy/helm/kubelm -n kubelm --create-namespace \
  --set k8sgpt.enabled=true --set agent.enabled=true
```

Pick a tier by cluster resources (see `values.yaml`): ultra-edge
(0.8B), edge (1.5B), edge+ (2B, default). **Every tier fits a 4 GB
node** — size for CPU/latency, not RAM. Full walkthrough — install,
K8sGPT wiring, the turnkey loop, air-gapped, NetworkPolicy — in
[`docs/deploying-kubelm-with-k8sgpt.md`](../../../docs/deploying-kubelm-with-k8sgpt.md).

| key | default | purpose |
|---|---|---|
| `model.hfRepo` / `model.hfFile` | `rbentaarit/kubelm-qwen3.5-2b-v1` | GGUF pulled at pod start |
| `model.localPath` | `""` | pre-seeded GGUF path (air-gapped); overrides hfRepo |
| `model.servedName` | `kubelm-qwen3.5-2b` | name K8sGPT references |
| `server.contextSize` | `16384` | llama-server `-c` |
| `resources` | req 2 CPU/2 Gi, lim 4 CPU/3 Gi | edge+; matches measured footprint |
| `networkPolicy.enabled` | `false` | restrict endpoint to K8sGPT |
| `cache.persistence.enabled` | `false` | PVC for the model (survives restarts / air-gap) |
| `k8sgpt.enabled` | `false` | deploy K8sGPT MCP server in-cluster (turnkey) |
| `agent.enabled` | `false` | deploy the agent loop (`POST /investigate`) |
