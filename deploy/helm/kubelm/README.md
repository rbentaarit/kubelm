# kubelm Helm chart

Deploys a kubelm tool-use model behind an OpenAI-compatible endpoint
(llama.cpp `llama-server`, CPU-only) for K8sGPT to use as its LLM
backend. Pinned to K8sGPT **v0.4.32**.

```bash
helm install kubelm deploy/helm/kubelm -n kubelm --create-namespace
```

Pick a tier by cluster resources (see `values.yaml`): ultra-edge
(0.8B), edge (1.5B), edge+ (2B, default). Full walkthrough — install,
K8sGPT wiring, air-gapped, NetworkPolicy — in
[`docs/deploying-kubelm-with-k8sgpt.md`](../../../docs/deploying-kubelm-with-k8sgpt.md).

| key | default | purpose |
|---|---|---|
| `model.hfRepo` / `model.hfFile` | `rbentaarit/kubelm-edge-v0.3-GGUF` | GGUF pulled at pod start |
| `model.localPath` | `""` | pre-seeded GGUF path (air-gapped); overrides hfRepo |
| `model.servedName` | `kubelm-edge` | name K8sGPT references |
| `server.contextSize` | `16384` | llama-server `-c` |
| `resources` | edge+ (6–8 Gi) | match the chosen tier |
| `networkPolicy.enabled` | `false` | restrict endpoint to K8sGPT |
| `cache.persistence.enabled` | `false` | PVC for the model (survives restarts / air-gap) |
