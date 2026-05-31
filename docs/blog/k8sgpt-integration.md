# Deploying kubelm alongside K8sGPT: a CPU-only tool-use model in your cluster

*Draft. Phase 6 — turning a trained, evaluated model into a deployable
artifact wired to K8sGPT.*

Everything before this post was measurement: train a small model on
multi-step K8sGPT-MCP trajectories, score it in a harness against a
library of deterministic failing clusters, publish the numbers. None of
that is usable by anyone running K8sGPT in a real cluster. Phase 6
closes that gap with a Helm chart that deploys kubelm **CPU-only**
behind an OpenAI-compatible endpoint, so K8sGPT can call it as its LLM
backend.

The design constraint that shapes everything here: kubelm proposes, it
does not execute. K8sGPT's MCP surface stays canonical — kubelm is
simply the model K8sGPT drives. Destructive actions remain gated by
K8sGPT's operator (Mutation CRs + policy), never by the model. This
post assumes you've read [PROJECT.md](../../PROJECT.md) and
[ROADMAP.md](../../ROADMAP.md) for what kubelm is and isn't.

---

## The architecture

Three pieces, all already in the K8sGPT ecosystem:

1. **An inference server** — llama.cpp's `llama-server` hosting the GGUF,
   exposing the de-facto-standard OpenAI `/v1` API. CPU-only; no GPU in
   the runtime path.
2. **A Service** exposing that endpoint inside the cluster.
3. **K8sGPT** pointed at the Service as its OpenAI-compatible backend.

```
K8sGPT  ──MCP tool-use──▶  kubelm (llama-server /v1)  ──▶  conclusion
   │                                                          │
   └────────── proposes; operator gates & disposes ◀──────────┘
```

The chart (`deploy/helm/kubelm/`) deploys (1) and (2). It pulls the
GGUF straight from Hugging Face at pod start via llama.cpp's
`--hf-repo`/`--hf-file`, or from a pre-seeded PVC for air-gapped
clusters. An optional NetworkPolicy restricts the endpoint to K8sGPT
pods only — kubelm is not a general-purpose LLM service you want
exposed cluster-wide.

```bash
helm install kubelm deploy/helm/kubelm -n kubelm --create-namespace
```

---

## One CPU-only family across a resource spectrum

The chart's `values.yaml` is where the deployment story lives, and it is
*not* "one best model with weaker fallbacks." kubelm is a single
CPU-only family spanning a resource spectrum — you pick the model that
fits the hardware, from the smallest local box up to a dev machine.
Each tier is judged on fitness for *its own* resource bracket, not
against the tier above.

| tier | model | GGUF / RAM | rubric¹ | CPU step² |
|---|---|---|---|---|
| ultra-edge | Qwen3.5-0.8B | 517 MB / 2–3 GB | 24/35 | ~5.5 s |
| edge | Qwen2.5-1.5B (v0) | 940 MB / 4 GB | 29/35 | ~9.6 s |
| edge+ *(default)* | Qwen3.5-2B (v0.3) | 1.2 GB / 8 GB | 32/35 | ~8.7 s |

¹ Conclusion-rubric pass rate on the 35-scenario library.
² Estimated cached per-step latency, CPU-only on an M1 Max — an *upper
bound*; commodity cluster CPUs run several× slower.

A surprise from measuring the CPU latency directly (first time we did,
prior benches all ran GPU-offloaded): the 2B (Qwen3.5, hybrid
linear-attention) processes prompts *faster* than the 1.5B (Qwen2.5,
dense) despite more parameters — linear-attention layers are cheaper on
the large tool-schema prompts K8sGPT sends. The ladder is not strictly
monotonic in parameter count.

---

## Two things the chart gets right that aren't obvious

**Serve no-think by default.** The kubelm models are trained to conclude
without `<think>` scaffolding (the deployment behavior). But K8sGPT does
not send the `enable_thinking` chat-template kwarg itself — so the
*server* must default to it. The chart passes
`--chat-template-kwargs '{"enable_thinking": false}'`. Leaving thinking
on produces longer, slower generations, which on CPU is the difference
between a usable step and a timeout.

**Size the context window generously.** A multi-step investigation
accumulates context — tool schemas, prior calls, and K8sGPT's
several-KB tool results. If the window is too small, a long
investigation doesn't degrade gracefully; it returns an HTTP 400 mid-run
and the whole trajectory is lost. The chart defaults `-c` to the model's
training `max_seq_length` (16384 for v0.3). Match or exceed it.

---

## Does it actually work end-to-end? (2026-05-29)

The faithful test for kubelm is the multi-step MCP loop — the model
driving K8sGPT's tools to a root cause — not a single-shot "explain this
finding" call. So: deploy the chart into a kind cluster, then run the
eval harness (the proven MCP loop driver) against a seeded scenario,
pointed at the **chart-served** endpoint.

Result on `configmap-missing-001`, chart-deployed v0.3, CPU, no-think:

```
label:     complete
schema:    2/2 valid   name_halluc=0   arg_halluc=0
refcalls:  passed=True
rubric:    passed=True   missing=[]   forbidden=[]
latency:   model=190852ms   steps=4    (~48 s/step)
```

The deployed model drove the investigation to the correct root cause.
That is the entire thesis — a small CPU-only model doing reliable
tool-use against K8sGPT — demonstrated as a running artifact rather than
a benchmark row.

It also took **~48 s per step** on a 6-core kind pod. That number is the
real story of CPU deployment: a 2B on constrained, contended CPU is
slow, and the first attempts *timed out* against the harness's hardcoded
120 s per-call limit. (We made that timeout configurable in the
process — it was too short for any CPU-served model.) This is exactly
why the ultra-edge 0.8B tier exists: at roughly half the per-step cost,
it is the right tool when the box is small and CPU-bound, even though
its rubric trails the 2B. Pick by hardware.

---

## Caveats (please read before deploying)

- **Latency is the binding constraint, not accuracy — and it swings
  ~10× by node.** RAM is not the gate (every tier fits ~2 GB); CPU is.
  But per-step latency depends far more on the *host* than the tier:
  two same-spec cloud CPU nodes measured ~7–10× apart (a fast modern
  x86 vs a throttled/oversubscribed one). The published figures are a
  *dedicated-vCPU reference*, not a guarantee — give the pod
  **guaranteed** cores, prefer a smaller tier, and expect multiples on
  burstable/contended nodes. K8sGPT investigations are not
  interactive-chat latency.
- **Not tested on a managed cluster.** The end-to-end validation is on
  kind. EKS/GKE/AKS is deferred; the integration contract is
  cluster-agnostic but a managed-cluster pass is unproven.
- **ollama is not supported for the Qwen3.5 tiers.** ollama 0.23.1's
  `qwen3next` loader rejects the v0.3/0.8B GGUFs; llama.cpp loads them
  fine. The v0 (1.5B) tier works under either. The chart uses
  llama-server throughout.
- **The 0.8B is unreleased.** It's a validated local artifact, not yet
  on Hugging Face — the chart defaults to v0.3 (edge+). The 0.8B row
  above is from local evaluation.
- **K8sGPT version pin.** kubelm was trained and evaluated against
  K8sGPT v0.4.32. Other versions are untested.

---

## Try it

Full walkthrough — install, K8sGPT wiring, air-gapped, NetworkPolicy —
is in [`docs/deploying-kubelm-with-k8sgpt.md`](../deploying-kubelm-with-k8sgpt.md).
The chart is `deploy/helm/kubelm/`. CPU latency data backing the tier
table is `eval/results/summaries/cpu-latency-2026-05-29.json`.
