# kubelm

[![CI](https://github.com/rbentaarit/kubelm/actions/workflows/ci.yml/badge.svg)](https://github.com/rbentaarit/kubelm/actions/workflows/ci.yml)

**A small, CPU-only language model specialized for reliable tool-use against
K8sGPT's MCP server.**

`kubelm` is an open project to build a family of small (0.8B–3B) language models
fine-tuned for one job: reliably and accurately using
[K8sGPT](https://k8sgpt.ai)'s Model Context Protocol (MCP) tools without
hallucinating tool names, fabricating arguments, or inventing cluster state.
The models target commodity CPU hardware — no GPU required — and follow
K8sGPT's MCP surface as it evolves.

> **Status:** Working. The eval harness + scenario library are shipped, the
> baseline benchmark is published, and two models are released on Hugging
> Face — **`kubelm-qwen3.5-2b-v1`** (Qwen3.5-2B) is the current headline and
> beats `qwen2.5:7b` on every reliability metric at ~⅓ the footprint. A
> **turnkey Helm chart** deploys kubelm + K8sGPT + an MCP agent loop in one
> `helm install`. See the tier ladder and phase status below.

---

## Why this exists

K8sGPT exposes Kubernetes operations as MCP tools — a standardized interface
that lets language models investigate clusters, query resources, fetch logs
and events, and propose remediation. The K8sGPT team explicitly recommends
local backends for production environments.

Today, that recommendation has two practical paths:

- **Run a large generic local model** (Llama 3.3 70B, Qwen 2.5 32B). Good at
  tool-use, but requires substantial GPU resources — typically 40GB+ VRAM —
  which most clusters don't have provisioned for AI workloads.
- **Run a small generic local model** (Llama 3.2 3B, Qwen 2.5 3B). Fits on
  commodity hardware, but routinely fails on tool-use: invents tool names,
  malforms arguments, fabricates cluster state, or fails to terminate.

There is no small local model that's *reliable* at K8sGPT's MCP tools. That
is the gap `kubelm` aims to fill.

The hypothesis: a small model fine-tuned specifically on K8sGPT MCP
trajectories — correct tool selection, well-formed arguments, faithful
grounding in tool results, sensible termination — can match the tool-use
reliability of much larger generic models on this specific surface, while
running on a single CPU node.

---

## What kubelm is and isn't

**kubelm is** a tool-use specialist for K8sGPT. The deliverables are:

- A reproducible evaluation harness that runs models as MCP clients against
  K8sGPT's real MCP server, measuring tool-use reliability metrics.
- A public benchmark comparing frontier cloud models, large local models, and
  small local models on K8sGPT MCP tool-use.
- Trajectory-based training datasets, public on Hugging Face.
- A family of fine-tuned small models, released open-weight, optimized for
  K8sGPT MCP tool-use on CPU.
- A Helm-deployable, CPU-only inference stack K8sGPT can use as a local
  backend — plus an optional turnkey loop (bundled K8sGPT MCP server + an
  agent that drives kubelm through its tools) so one `helm install` makes a
  cluster queryable end-to-end.

**kubelm is not:**

- **Not a fork of K8sGPT.** Integrates via existing local-backend interfaces
  (`customrest`, `ollama`, `localai`). K8sGPT remains the analyzer,
  orchestrator, and MCP server.
- **Not a custom tool surface.** Tracks K8sGPT's MCP surface as it evolves.
  No divergent tools, no parallel ecosystem.
- **Not a remediation engine.** K8sGPT's auto-remediation architecture
  (Mutation CRs, operator policy gates, thresholds, rollback) handles
  destructive actions. kubelm proposes; the operator gates and disposes.
- **Not a frontier-model replacement.** For users with no
  cost/latency/hardware constraints, GPT-4-class cloud models will likely
  remain higher quality on the long tail. kubelm targets the cases where
  running locally on small hardware matters.
- **Not a research project.** Goal is shippable infrastructure using
  well-established techniques. Innovation is in the training data, the
  evaluation methodology, and the deployment story — not the training
  algorithm.

---

## What "reliable tool-use" actually means

The benchmark measures specific failure modes that distinguish small models
from large ones. These are the headline metrics:

- **Tool-name hallucination rate.** Frequency of calls to nonexistent tools
  or misspelled tool names.
- **Argument hallucination rate.** Frequency of arguments that don't match
  the tool's schema (wrong types, invented fields, missing required fields).
- **Grounding failure rate.** Frequency of conclusions that reference cluster
  state never returned by any tool call.
- **Termination failure rate.** Frequency of trajectories that loop, don't
  reach a conclusion, or terminate prematurely.
- **Task completion rate.** End-to-end: did the trajectory reach a correct
  conclusion for the seeded scenario?

A specialized small model that matches large-model task-completion *and* has
materially lower hallucination rates is the target outcome. That combination
is what justifies the project's existence.

---

## Project goals

1. **Publish a public K8sGPT MCP tool-use benchmark.** Reproducible eval
   harness running real K8sGPT MCP servers against real clusters with
   seeded scenarios.

2. **Build trajectory-based training datasets** anchored to K8sGPT's MCP
   surface, public on Hugging Face.

3. **Train and release a tiered family of specialized models.** Three sizes
   targeting different cluster resource profiles. Same training methodology,
   different base sizes.

4. **Integrate with K8sGPT as a drop-in local backend.** No fork. Use the
   existing OpenAI-compatible local-backend interfaces.

5. **Ship as cluster-native infrastructure.** A Helm chart that deploys the
   model and inference engine alongside K8sGPT, with sensible defaults,
   sizing guidance, and no GPU dependencies.

---

## The tier ladder

kubelm is **one CPU-only family across a resource spectrum** — not one best
model with weaker fallbacks. Pick the model that fits your hardware, from the
smallest local box up to a dev machine; each tier is judged on fitness for
*its own* resource bracket. All numbers below are **measured**, not targets.

| Tier | Model | Rubric¹ | Serving RAM² | Per-step³ | Release |
|---|---|---|---|---|---|
| ultra-edge | `kubelm-qwen3.5-0.8b-v1` | 24/35 | ~0.9 GB | ~16–32 s | fine-tuned; not yet on HF |
| edge | `kubelm-qwen2.5-1.5b-v1` | 29/35 | ~1.1 GB | ~20–40 s | on Hugging Face |
| **edge+** *(default)* | `kubelm-qwen3.5-2b-v1` | **32/35** | ~1.6 GB | ~29–55 s | on Hugging Face |
| standard | ~3B | — | — | — | planned |

*For reference, the `qwen2.5:7b` local model scores rubric 28/35 on the same
suite — so `kubelm-qwen3.5-2b-v1` beats a 7B generic model at ~⅓ the footprint,
and even the 0.8B is competitive.*

¹ Conclusion-rubric pass rate over the 35-scenario library (does the model
reach the correct root cause, grounded in tool results). All tiers are
schema-perfect with **zero tool-name/argument hallucinations**.
² True serving footprint (`--no-mmap` RSS at `-c 16384`), measured on real
x86 Linux. The hybrid linear-attention KV cache is compact — **every tier
fits a 4 GB node; RAM is not the binding constraint, CPU/latency is.**
³ Per investigation step on a **dedicated** x86 Linux 2-core node (typical →
heavy prompt); a full investigation is ~1–4 min. This is a dedicated-vCPU
reference, **not a guarantee — latency swings ~10× by host**, so give the
pod guaranteed cores. Full data: `eval/results/summaries/cpu-latency-2026-05-29.json`.

---

## Methodology

Tool-use reliability is determined by methodology, not by training scale. The
methodology this project commits to:

- **Eval-first.** The benchmark is built before any fine-tuning. Every
  dataset addition, every training run, every model release is measured
  against the same eval suite. No vibe-based progress claims.
- **Tested against the real K8sGPT MCP server.** No mocks for the eval. We
  boot K8sGPT against real (kind) clusters with seeded failure scenarios
  and run models as MCP clients. Results reflect actual deployed behavior.
- **K8sGPT MCP surface is canonical.** kubelm tracks K8sGPT's tool surface
  as it evolves. Every model release is pinned to a specific K8sGPT version.
  No divergence.
- **Trajectory-based training data.** Examples are multi-step
  (goal → tool call → result → ... → conclusion), not diagnostic prose.
  This trains the behavior we actually care about.
- **Reproducible training.** Datasets, training scripts, and hyperparameters
  are all public. Anyone can re-run the training and verify the results.
- **Open-weight releases.** All models are Apache 2.0, weights on Hugging
  Face. No closed weights, no custom licenses, no gated downloads.

---

## Safety model

K8sGPT's architecture handles destructive operations through Mutation Custom
Resources, operator policy gates, configurable risk thresholds, and rollback
mechanisms. The *system* gates destructive actions; the model proposes them.

kubelm follows this. The model is not trained to make safety decisions for
destructive operations — that is the operator's job. The model is trained
for reliability properties: correct tool calls, faithful grounding,
appropriate termination, structured output that's consumable by the
operator's policy layer.

This separation of concerns is intentional and matches K8sGPT's existing
design.

---

## What's in this repo (and what isn't, yet)

This repo grows in stages. Each stage is a separately-useful artifact:

- [x] **Phase 1: Eval harness skeleton** — runs a model as an MCP client
      against a real K8sGPT MCP server; records trajectories, scores
      reliability metrics. *Shipped.*
- [x] **Phase 2: Seeded scenario library** — 35 kind-based scenarios paired
      with reference trajectories: pod-startup, service/networking,
      scheduling, storage, RBAC, resources, workload controllers,
      multi-hop. *Shipped.*
- [x] **Phase 3: Public baseline benchmark** — cloud frontier (`gpt-5.4`),
      large local (`qwen2.5:32b`), and small local models measured against
      the eval. *Shipped; cuts in `eval/results/summaries/`.*
- [x] **Phase 4: Trajectory training dataset** — multi-step examples on
      Hugging Face (`kubelm-seed`), pinned to K8sGPT v0.4.32. *Shipped.*
- [x] **Phase 5: Fine-tuned model releases** — `kubelm-qwen2.5-1.5b-v1` and
      `kubelm-qwen3.5-2b-v1` (the headline) on Hugging Face; the 2B beats
      `qwen2.5:7b` on every metric (rubric 32 vs 28, fabrications 3 vs 8)
      at ~⅓ the footprint. A Qwen3.5-0.8B ultra-edge tier is fine-tuned
      (rubric 24, 517 MB) but not yet released. *Shipped.*
- [x] **Phase 6: K8sGPT integration** — Helm chart (`deploy/helm/kubelm/`)
      that deploys kubelm CPU-only behind an OpenAI endpoint, plus an
      optional **turnkey loop** (K8sGPT MCP server + an agent that drives
      kubelm through its tools). Validated end-to-end on kind; agent image
      published multi-arch to GHCR. *Shipped.*
- [ ] **Phase 7: Model ladder expansion** — a `standard` (~3B) tier, and a
      possible 0.8B release, evaluated against the same benchmark.

Each phase shipped publicly before the next began.

---

## How to follow along

- **GitHub:** This repo is the source of truth for code, datasets, and
  benchmark results.
- **Hugging Face:** [`kubelm-qwen3.5-2b-v1`](https://huggingface.co/rbentaarit/kubelm-qwen3.5-2b-v1)
  (current headline) and [`kubelm-qwen2.5-1.5b-v1`](https://huggingface.co/rbentaarit/kubelm-qwen2.5-1.5b-v1)
  models; the [`kubelm-seed`](https://huggingface.co/datasets/rbentaarit/kubelm-seed)
  trajectory dataset.
- **Blog posts:** Major milestones will be written up. Links added here as
  they're published.

---

## Contributing

The harness, the 35-scenario library, the baseline benchmark, the
released models, and the Helm chart are all shipped. The most useful
contribution today is still **additional K8sGPT MCP tool-use
scenarios**: real Kubernetes investigation flows you've encountered,
with the sequence of tool calls a competent SRE would make. Broader
scenario coverage improves the eval for every tier and sharpens the
next training cycle.

See `eval/scenarios/specs/` for the format and existing examples. Each
scenario is a YAML spec with a kind setup, an investigation goal, a
list of acceptable reference tool calls (`any_of`), and a substring
conclusion rubric. New scenarios should be validated end-to-end via
`uv run python -m eval.scenarios run --scenario-id <id> ...` before
submission.

---

## Relationship to K8sGPT

`kubelm` exists because of [K8sGPT](https://github.com/k8sgpt-ai/k8sgpt),
not in spite of it. K8sGPT provides the analyzer framework, the
orchestration, the cluster integration, the MCP server, and the
auto-remediation architecture. `kubelm` aims to be the model that makes
K8sGPT's recommended-for-production local-backend path actually reliable
at tool-use on small hardware.

This project is not affiliated with or endorsed by the K8sGPT project.
"K8sGPT" is referenced for compatibility purposes only.

---

## License

- **Code:** Apache 2.0
- **Models:** Apache 2.0
- **Dataset:** CC BY 4.0

---

## A note on framing

This project is built in public for two reasons. First, the artifacts
(benchmark, datasets, methodology) are valuable independent of whether the
models themselves succeed. Second, building in public forces methodological
honesty: every claim is reproducible, every result is comparable.

If the core hypothesis turns out to be wrong — if generic small models are
already reliable enough at K8sGPT MCP tool-use, or if a curated prompt
template alone closes the gap — that's a publishable result too, and it
changes the project rather than ending it. The eval harness and scenario
library remain useful regardless.

The work happens at the intersection of three communities: K8s/CNCF,
applied ML, and platform engineering. Discussions, corrections, and
collaboration from any of these are welcome.
