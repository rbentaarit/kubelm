# Deploying kubelm with K8sGPT

This guide deploys a kubelm tool-use model into a Kubernetes cluster and
wires K8sGPT to use it as the LLM backend. kubelm runs **CPU-only** and
exposes an **OpenAI-compatible** endpoint via llama.cpp's `llama-server`.
K8sGPT's MCP surface stays canonical — kubelm is simply the model it
calls.

Pinned to **K8sGPT v0.4.32** (the version kubelm was trained and
evaluated against). Using a different K8sGPT version is untested.

## 1. Pick a tier

kubelm is one CPU-only family across a resource spectrum. Pick the
model that fits the cluster; each tier is the right-sized tool-use
model for its bracket, not a downgrade of the one above.

| tier | model | serving RAM¹ | rubric | step @2-core x86¹ | HF repo |
|---|---|---|---|---|---|
| ultra-edge | Qwen3.5-0.8B | ~0.9 GB | 24/35 | ~16–32 s | `rbentaarit/kubelm-qwen3.5-0.8b-v1` |
| edge | Qwen2.5-1.5B | ~1.1 GB | 29/35 | ~20–40 s | `rbentaarit/kubelm-qwen2.5-1.5b-v1` |
| **edge+** *(default)* | Qwen3.5-2B | ~1.6 GB | 32/35 | ~29–55 s | `rbentaarit/kubelm-qwen3.5-2b-v1` |

¹ Serving footprint is measured (`--no-mmap` RSS, host-independent).
Per-step latency is measured on a **dedicated** x86 Linux 2-core node
(`-ngl 0`); full investigation ~1–4 min. The 1.5B latency is
**estimated**. Data: `eval/results/summaries/cpu-latency-2026-05-29.json`.

> **⚠️ Latency is host-dependent (~10×), and that dominates the tier
> choice.** The figures above are a *dedicated-vCPU reference*, not a
> guarantee — two same-spec cloud CPU nodes measured ~7–10× apart (a
> fast modern x86 vs a throttled/oversubscribed host). **Give the pod
> guaranteed (not burstable/shared) CPU**, and expect multiples of the
> reference latency on contended nodes. RAM is unaffected.

**RAM is not the gate** — every tier fits a 4 GB node (compact hybrid
KV cache). The chart defaults (`requests` 2 CPU/2 Gi, `limits` 4 CPU/3
Gi) suit edge+; drop CPU to 2 for the smallest nodes, or pick the
ultra-edge tier for ~½ the per-step latency.

## 2. Install the chart

```bash
helm install kubelm deploy/helm/kubelm \
  --namespace kubelm --create-namespace
```

First start pulls the GGUF from Hugging Face (the `startupProbe` allows
several minutes). Watch it come up:

```bash
kubectl -n kubelm rollout status deploy/kubelm
kubectl -n kubelm port-forward svc/kubelm 8080:8080
curl http://127.0.0.1:8080/v1/models      # should list the served model
```

### Air-gapped / no egress

Pre-seed the GGUF onto a PersistentVolume and point the chart at it:

```bash
helm install kubelm deploy/helm/kubelm -n kubelm --create-namespace \
  --set cache.persistence.enabled=true \
  --set model.hfRepo="" \
  --set model.localPath=/cache/kubelm-qwen3.5-2b-v1.Q4_K_M.gguf
```

(Copy the GGUF into the PVC out-of-band, e.g. a one-shot loader Job.)

## 3. Wire K8sGPT to the endpoint

Point K8sGPT's OpenAI-compatible backend at the in-cluster Service:

```bash
k8sgpt auth add --backend customrest \
  --baseurl http://kubelm.kubelm.svc:8080/v1 \
  --model kubelm-qwen3.5-2b
k8sgpt analyze --explain --backend customrest
```

The served model name (`--model`) must match `model.servedName` in
values (default `kubelm-qwen3.5-2b`).

## 4. Restrict access (shared clusters)

By default any pod in the namespace can reach the endpoint. To limit it
to K8sGPT only:

```bash
helm upgrade kubelm deploy/helm/kubelm -n kubelm \
  --set networkPolicy.enabled=true
```

Adjust `networkPolicy.k8sgptPodSelector` to match your K8sGPT pod labels
(requires a NetworkPolicy-enforcing CNI, e.g. Calico/Cilium).

## 5. Turnkey: full in-cluster loop (K8sGPT + agent)

Sections 1–4 deploy the **model server** and expect you to bring your own
K8sGPT. To get a self-contained "submit a goal, kubelm investigates"
install, enable the bundled K8sGPT MCP server and the **agent** — a thin
service that drives kubelm through K8sGPT's MCP tools (it reuses the eval
run-loop; it reimplements nothing).

The agent image is published multi-arch (amd64 + arm64) at
`ghcr.io/rbentaarit/kubelm-agent`, so this is a plain install:

```bash
helm install kubelm deploy/helm/kubelm -n kubelm --create-namespace \
  --set k8sgpt.enabled=true --set agent.enabled=true
```

For local development of the agent, build and override the image:

```bash
docker build -f deploy/agent/Dockerfile -t kubelm-agent:dev .
kind load docker-image kubelm-agent:dev --name <cluster>
helm install kubelm deploy/helm/kubelm -n kubelm --create-namespace \
  --set k8sgpt.enabled=true --set agent.enabled=true \
  --set agent.image.repository=kubelm-agent --set agent.image.tag=dev
```

This adds: K8sGPT (`serve --mcp`, pinned v0.4.32, read-only ClusterRole)
and the agent (`POST /investigate`). Query it:

```bash
kubectl -n kubelm port-forward svc/kubelm-agent 8088:8080
curl -s http://127.0.0.1:8088/investigate \
  -H 'Content-Type: application/json' \
  -d '{"goal":"Why is the web deployment in namespace prod not starting?"}'
# -> {"conclusion": "...", "termination": "complete", "steps": N,
#     "tool_calls": [{"name":"get-resource", ...}]}
```

Notes:
- The agent's per-step timeout (`agent.requestTimeout`, default 600 s)
  must comfortably exceed kubelm's CPU per-step latency — see the
  envelope summary. Investigations take ~1–4 min on a dedicated node.
- K8sGPT's `serve` requires an AI provider at startup; the chart points
  its backend at kubelm to satisfy that. The agent drives the tools
  regardless.

## Notes

- **Why llama-server, not ollama:** the Qwen3.5 tiers (0.8B, v0.3) load
  cleanly under llama.cpp but are rejected by ollama 0.23.1's
  `qwen3next` loader. The v0 (1.5B) tier works under either.
- **Context window (`-c`):** defaults to 16384 (v0.3's training
  `max_seq_length`). A too-small window turns a long multi-step
  investigation into an HTTP 400 rather than a `no_conclusion`; size it
  generously. See the bench serving-validity note in PROJECT.md.
- **Safety:** kubelm proposes; it does not execute. Destructive actions
  are gated by K8sGPT's operator (Mutation CRs + policy), not by the
  model.
