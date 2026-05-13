# ROADMAP.md

Phased plan for kubelm. Each phase produces an independently valuable
artifact. Phase order is firm; specific deliverables within a phase may
evolve as data comes in.

This is a living document.

---

## Phase 0: Foundation (DONE)

- [x] Public GitHub repo: github.com/rbentaarit/kubelm
- [x] README.md
- [x] LICENSE (Apache 2.0)
- [x] PROJECT.md (thesis and methodology)
- [x] ROADMAP.md (this file)
- [x] .gitignore
- [x] CLAUDE.md (AI-assistant instructions)

---

## Phase 1: Eval Harness Skeleton

**Goal:** Python framework that runs a model as an MCP client against a
real K8sGPT MCP server, records the full trajectory, and computes
reliability metrics.

**Deliverable:** `eval/` directory with:

- MCP client implementation (HTTP-based against K8sGPT's MCP server, not
  stdin/stdout — easier to control and instrument)
- Trajectory recorder (full JSONL log of every model output, every tool
  call, every tool result, every conclusion)
- Metric calculators:
  - Tool-name hallucination rate
  - Argument-schema validation rate (against K8sGPT's tool schemas)
  - Grounding analyzer (does the conclusion reference state never returned
    by tools?)
  - Termination classifier
  - Per-step latency tracker
- Pluggable model backends: any OpenAI-compatible endpoint, plus direct
  Ollama and llama.cpp integrations
- CLI to run a model + scenario combination, output a results JSON

**Key design principles:**

- Tested against the real K8sGPT MCP server — never against a mock.
- Schema validation is automated against K8sGPT's actual tool schemas
  (fetched via `tools/list`).
- Results are machine-comparable JSON for diff-friendly tracking across
  model versions.

**Tooling notes:**

- Python with simple dependencies: `requests` for HTTP MCP, `jsonschema`
  for argument validation, `pytest` for harness tests.
- Avoid heavy ML frameworks at this stage — the eval doesn't need them.

### Phase 1 checklist

- [x] MCP HTTP client implementation (initialize, tools/list, tools/call)
- [x] Trajectory recorder format defined (JSONL schema)
- [x] Argument-schema validator
- [x] Grounding analyzer (rule-based; LLM-judge variant as v0.2)
- [x] Termination classifier
- [x] CLI runner
- [x] First end-to-end test: run a model against K8sGPT, record trajectory

---

## Phase 2: Seeded Scenario Library

**Goal:** a library of kind-based test scenarios paired with reference
trajectories — what tool calls should a competent SRE make for each
failure?

**Deliverable:** `eval/scenarios/` with:

- 30–50 initial scenarios covering common K8s failure modes
- For each scenario: a kind cluster setup script (or Helm chart) that
  produces the failure, plus a reference trajectory describing the
  expected investigation pattern
- A scenario runner that brings up the cluster, points K8sGPT at it, and
  runs the eval against models

**Coverage targets for v0.1:**

- Common pod-startup failures (CrashLoopBackOff, ImagePullBackOff,
  init-container failures)
- Service connectivity issues (selector mismatches, NetworkPolicy blocks)
- Resource issues (OOMKilled, resource quota blocks)
- RBAC failures
- Storage failures (PVC binding)

**Reference trajectory format per scenario:**

- Goal statement (what the user is asking)
- Expected tool calls in order, with expected argument structure
- Acceptable variations (some scenarios have multiple valid investigation
  paths)
- Conclusion rubric (what facts must appear in a correct conclusion)

### Phase 2 checklist

- [x] Scenario format documented
- [x] First 10 scenarios with kind setup scripts and reference trajectories
- [x] Scenario runner integrated with eval harness
- [x] 30–50 scenarios covering top failure modes (30 as of 2026-05-11;
      categories: pod-startup, service/networking, scheduling, storage,
      RBAC, resources, workload controllers)
- [x] Scenario library publicly committed
- [x] Blog post draft on the scenario methodology

---

## Phase 3: Public Baseline Benchmark

**Goal:** run the eval against frontier cloud models, large local models,
and small generic local models. Publish results as the foundational
artifact.

**Deliverable:** `benchmark/` directory with results, plus a public blog
post.

**Models to benchmark (initial):**

- GPT-4 / GPT-4o (cloud baseline)
- Claude (cloud baseline)
- Llama 3.3 70B via Ollama (large local — what serious K8sGPT users
  actually run today)
- Qwen 2.5 32B via Ollama (alternative large local)
- Llama 3.2 3B via Ollama (small local — what `kubelm-standard` will
  compete against)
- Qwen 2.5 3B (alternative small local)
- Phi-3.5 mini (alternative small local)
- Qwen 2.5 1.5B (smallest local baseline)

**Hardware setup:**

- Cloud models: API access
- Large local models: rented GPU box (A100 or similar) for benchmarking
- Small local models: dedicated CPU box matching `kubelm-standard` target
  hardware (4-core, 4GB allocated)

**Headline result:** how do hallucination metrics scale with model size on
this surface? The expectation is a sharp falloff in tool-use reliability
below ~7B for generic models. If true, kubelm has a clear gap to fill.
If false (generic small models are already reliable), the project pivots.

**Blog post:** "Benchmarking 7+ LLMs on K8sGPT MCP tool-use: how
hallucination rates scale with model size."

### Phase 3 checklist

- [x] Benchmark plan documented (Shape A and Shape B model lineups in
      `eval/scenarios/benchmarks/`)
- [x] All benchmark runs completed (3 published cuts as of 2026-05-12:
      `shape-a-2026-05-07`, `shape-b-2026-05-11` (10 scenarios), and
      `shape-b-2026-05-12` (full 30-scenario library, post-retrofit).
      The 70B local GPU-box run was originally listed but dropped on
      2026-05-12 as not load-bearing for the decision gate — the
      qwen2.5:32b ≈ gpt-4o plateau in the 2026-05-12 cut is
      sufficient evidence that "above 7B is flat" extends to the
      frontier, so a 70B point would only confirm what's already
      visible. See PROJECT.md decisions log)
- [x] Results table with all reliability metrics + performance metrics
      (`eval/results/summaries/README.md`)
- [x] Hallucination-vs-size visualization
      (`eval/results/summaries/shape-b-2026-05-11.png`, regenerated
      via `eval/results/summaries/plot_shape_b.py`)
- [x] Blog post draft (`docs/blog/scenario-methodology.md`)
- [x] Per-scenario audit of the 2026-05-12 outlier (gpt-5.4's
      30/30 grounding failure). Audit overturned the headline
      reading: most flagged facts are present in tool output but
      rendered in YAML-path / quoted / dotted-status format the
      v1 grounding analyzer can't match. The blog and PROJECT.md
      decisions log were revised accordingly. Tracked as the
      grounding-metric-v2 followup item below.
- [ ] Blog post published
- [ ] Results announced on relevant channels

### Followups surfaced during Phase 3

- **Grounding-metric v2.** The v1 rule-based analyzer's substring
  matcher mishandles structured paraphrase (dotted paths, quote
  variants, YAML notation, string composition from primitives). It
  systematically penalizes verbose-but-faithful models. A v2 needs
  to tolerate these rephrasings — either by parsing the model's
  output for structural facts and matching against tool-result
  JSON paths, or by using an LLM-judge to verify faithfulness
  rather than substring presence. Until this lands, cross-model
  grounding comparisons aren't reliable enough to drive
  architectural claims. May fit better in Phase 4 / Phase 5
  prep, but flagging here as a Phase 3 surfaced issue.
- **K8sGPT MCP `networkpolicies` upstream.** File against
  k8sgpt-ai/k8sgpt to add networkpolicy support to
  `list-resources`. Currently the
  `eval/scenarios/specs/network-policy-block-001.yaml` rubric
  is relaxed because the type isn't exposed. Re-tighten the
  rubric when the type lands.

**Decision gate after Phase 3:** does the data support continuing to
fine-tuning?

- If small generic models are already reliable enough: pivot to "ship a
  curated prompt template" and re-evaluate the project's value.
- If a clear hallucination gap exists between small and large: proceed to
  Phase 4 (training data construction).

---

## Phase 4: Trajectory Training Dataset

**Goal:** expert-curated multi-step training examples (trajectories)
anchored to a specific K8sGPT MCP version. Public on Hugging Face.

**Deliverable:** Hugging Face dataset with full provenance and version
pinning.

**Trajectory format per example:**

- System prompt (role + tool descriptions for the K8sGPT MCP version)
- Goal statement
- Sequence of (assistant tool call, tool result) pairs
- Final assistant conclusion
- Provenance metadata (source, K8sGPT version, scenario reference)

**Construction process:**

1. Generate seed trajectories from the Phase 2 scenarios using a strong
   model (Claude / GPT-4) with careful prompting. Manual review of each
   trajectory for correctness.
2. Hand-write trajectories for failure modes the seed model gets wrong.
3. Negative examples: trajectories where the model attempted a bad call
   (hallucinated tool name, wrong arguments) and was corrected. These
   teach the model to recover.
4. Vary surface details (namespaces, resource names, label values) to
   teach generalization rather than memorization.

**Coverage targets:**

- ~50% of examples on common failure modes from Phase 2
- ~25% on multi-step investigations (3+ tool calls before conclusion)
- ~15% on cases where the obvious first tool call is wrong (teaches
  better tool selection)
- ~10% on cases requiring synthesis of multiple tool results

**Target volume:** 500–2,000 trajectories for v0.1. Quality over volume.

### Phase 4 checklist

- [x] Trajectory format documented (`data/seed/FORMAT.md`,
      schema_version 1, OpenAI tool-use messages shape)
- [x] Seed-trajectory generation pipeline working
      (`data/seed/convert.py` takes an eval bench summary and
      emits trajectory JSONL; `data/seed/snapshot_tools.py`
      captures the K8sGPT MCP tools/list once per version).
      First seed file: `data/seed/v0/gpt-5.4-2026-05-12.jsonl`
      (29 trajectories — gpt-5.4's clean rubric runs from the
      2026-05-12 Shape B cut). Tools cache still empty until
      the snapshot script is run.
- [x] Per-trajectory review of the 29 seed trajectories. Auto-review
      via `data/seed/review.py` applied REVIEW.md's hard rules
      (schema/termination/rubric) and a token-level grounding-artifact
      heuristic. Result: 29/29 `accepted`, all with
      `grounding_failed_v1_artifact: true`. Residual unmatched facts
      across all trajectories are derived states (`NotReady`,
      `non-zero`), reasonable string compositions (`http://<pod>:80/healthz`),
      tool-name self-references (`list-resources`), or the known
      K8sGPT MCP networkpolicies gap (`default-deny`,
      `ingress-blocking`). No genuine fabrications found.
- [~] First 100 hand-reviewed trajectories. 29 reviewed seeds plus
      290 variants (10 per seed) = 319 trajectories on disk. The
      variants share their source's review_status because they are
      mechanical surface-detail substitutions, but a tightening review
      can re-stamp them. Hand-written trajectories (Phase 4 construction
      step 2) are still the gap to 100+ NEW seeds.
- [x] Generalization variation pipeline (surface-detail randomization).
      `data/seed/vary.py` substitutes namespace + primary resource
      names from curated pools, deterministically per
      (scenario_id, variant_idx), consistent across system prompt /
      goal / tool_call args / tool_result content / final
      conclusion. Output:
      `data/seed/varied/v0/gpt-5.4-2026-05-12-varied.jsonl` (10
      variants × 29 seeds = 290 trajectories, ~7.8 MB, zero
      substitution leakage).
- [x] Negative examples included.
      `data/seed/synthesize_negatives.py` produces 46 negative
      trajectories at v0 from two injection patterns:
        - wrong_resource_type (17 trajectories): injects
          `list-resources(resourceType=<typo>)` before the real
          call; the K8sGPT-shape `unsupported resource type` error
          response is reproduced verbatim from the 2026-05-12
          drill-in observation; the assistant follows with a short
          recovery message before the original trajectory runs.
        - hallucinated_tool_name (29 trajectories): injects a
          `get-pod` call (a plausible non-existent tool); MCP
          returns `unknown tool: get-pod`; recovery message names
          the real `get-resource` shape.
      All marked `provenance.source: "negative_synthetic"` and
      `review_status: "unreviewed"` — synthetic data needs a human
      pass before training, especially the recovery prose.
      Output: `data/seed/varied/v0/negatives-2026-05-12.jsonl`
      (~1.3 MB).
- [ ] Hugging Face dataset published (v0.1, pinned to a K8sGPT version).
      Upload recipe documented in `data/seed/README.md`; deliberately
      not scripted — first publication is a one-shot maintainer
      action. The corpus on disk (data/seed/v0/ + data/seed/varied/v0/)
      is ready to ship.
- [x] Dataset card with methodology, license, intended use
      (`data/seed/DATASET_CARD.md`, HF-compatible front matter,
      links to FORMAT.md / REVIEW.md / repo. Will become the HF
      dataset's README.md at upload time).
- [x] Blog post on dataset construction
      (`docs/blog/trajectory-dataset-construction.md`). Covers
      the format/shape choices, reusing eval-bench output as
      seeds, the review heuristic + grounding-artifact audit,
      the variation pipeline, negative-example synthesis, and
      honest limitations. ~400 lines, same tempo as the scenario-
      methodology blog. Includes a "How to use it" section with
      HF `datasets` snippets.

### Followups surfaced during Phase 4 prep

- **TrajectoryRecorder should persist the tools/list it saw.** The
  eval harness currently calls `client.list_tools()` at run time
  but doesn't save the result alongside the trajectory. Adding a
  `tools` field to the meta event would let Phase 4's converter
  ingest the exact tool schemas the model saw during the run, and
  would remove the need for an out-of-band `data/seed/tools/*.json`
  cache. Small change; defer to a quiet moment between bench runs
  so it doesn't perturb in-flight results.

---

## Phase 5: First Fine-Tuned Model

**Goal:** release `kubelm-standard` (3B) on Hugging Face, with reproducible
training pipeline.

**Deliverable:** Hugging Face model with weights, model card, and training
code in this repo.

**Approach:**

1. Pick base model based on Phase 3 results. Tool-use behavior in the
   base model matters as much as raw capability — strong starting points
   include Qwen 2.5 3B, Llama 3.2 3B, Phi-3.5 mini.
2. Supervised fine-tuning (SFT) on trajectories using QLoRA. Single A100
   per training run on RunPod or Modal. Per-run cost: under $10.
3. Evaluate after each run against the Phase 1 eval harness on the
   Phase 2 scenarios. Track in a results log.
4. Quantize to GGUF (Q4_K_M baseline) using llama.cpp toolchain.
5. Release: LoRA adapter and merged GGUF formats both on Hugging Face.

**Quality bar for release:**

- Tool-name hallucination rate: lower than the base model on the eval.
- Argument hallucination rate: lower than the base model.
- Task completion rate: at least matching the base model.

If the model isn't measurably better than its base on hallucination
metrics, don't release. Iterate the data.

### Phase 5 checklist

- [x] Base model selected based on Phase 3 data + a same-day
      2026-05-13 baseline measurement of the candidate:
      `Qwen/Qwen2.5-1.5B-Instruct` for `kubelm-edge` v0. The
      original 2026-05-13 choice of `Qwen/Qwen2.5-3B-Instruct`
      (kubelm-standard v0) was revised later the same day — the
      deployment story is "standalone / dev cluster" which argues
      for the 1.5B edge tier as v0. 2026-05-13 baseline
      (`eval/results/summaries/shape-b-2026-05-13-qwen-1.5b.json`):
      8/30 complete, 10/30 rubric, 3/30 ref_pass, 0 name
      hallucinations, 2 arg hallucinations — a real SFT
      foothold. HF survey for K8s-specialized small models came
      up empty for this surface (sub-2B candidates are
      smaller-base / wrong-surface / low-adoption). Full
      rationale in PROJECT.md decisions log 2026-05-13.
- [x] Training scaffolding committed (NOT a training run):
      `training/configs/kubelm-edge-v0.yaml` (Unsloth QLoRA —
      base model Qwen 2.5 1.5B, dataset filter, LoRA rank 32,
      3 epochs, lr 2e-4, per_device_batch 8, paged AdamW 8-bit),
      `training/sft.py` (entry point; deferred heavy imports
      so `--dry-run` works without CUDA),
      `training/eval_checkpoint.py` (bench adapter — boots a
      llama_cpp server OR points at an existing OpenAI-compat
      backend, then runs the standard 30-scenario Shape B
      against it for direct comparison to the baseline rows),
      `training/README.md` (orientation + deployment footprint +
      cost model + how-to). Dry-run loads 319 records (29 seeds
      + 290 variants; negatives excluded for v0).
- [ ] First training run completed end-to-end (rented A100;
      cost est. <$10)
- [ ] Hyperparameter sweep (5–10 runs)
- [ ] Best checkpoint selected via eval
- [ ] Quantized to GGUF
- [ ] Hugging Face release (v0.1, pinned to a K8sGPT version)
- [ ] Model card with eval results, intended use, limitations
- [ ] Blog post on the fine-tuning process and results

---

## Phase 6: K8sGPT Integration

**Goal:** Helm chart that deploys kubelm + inference engine alongside
K8sGPT in a real cluster.

**Deliverable:** `deploy/helm/kubelm/` chart, plus deployment guide.

**Architecture:**

- Inference server (llama.cpp server, vLLM, or Ollama) hosting the model
- Service exposing OpenAI-compatible endpoint inside the cluster
- K8sGPT configured to use this internal endpoint as its `customrest`,
  `ollama`, or `localai` backend
- Optional: NetworkPolicy restricting model access to K8sGPT only

**Sizing guidance documented:**

- Tier 1: edge / dev (when `kubelm-edge` ships)
- Tier 2: production default with `kubelm-standard` (4 cores, 4GB)
- Tier 3: large / regulated (when `kubelm-pro` ships)

**Test:** deploy to kind + a real managed K8s cluster (EKS / GKE / AKS).
Verify K8sGPT correctly routes to kubelm and the integration produces
sensible end-to-end behavior on the eval scenarios.

### Phase 6 checklist

- [ ] Helm chart skeleton
- [ ] Inference server deployment working
- [ ] K8sGPT integration tested end-to-end on kind
- [ ] Tested on a managed K8s cluster
- [ ] Sizing guidance documented
- [ ] Deployment guide in `docs/`
- [ ] Demo screencast (optional but high-value)
- [ ] Blog post on the integration

---

## Phase 7: Model Ladder Expansion

**Goal:** release `kubelm-standard` (3B) and `kubelm-pro` (7–8B)
variants, completing the ladder after `kubelm-edge` (1.5B) shipped
in Phase 5.

**Deliverable:** Two additional Hugging Face model releases, evaluation
results across the full three-tier ladder.

**Strategy notes:**

- Same dataset, same training recipe, different base models. Don't
  customize per tier — homogeneity keeps maintenance manageable.
- Re-run full eval against all three tiers. Publish comparison.
- Update Helm chart to support tier selection via `values.yaml`.
- Edge-first ordering (decided 2026-05-13): the deployment story
  is "K8sGPT alongside a small model in a standalone / dev
  cluster" → kubelm-edge ships first as v0 (Phase 5). kubelm-
  standard and kubelm-pro are the "more capability, more memory"
  variants that follow.

### Phase 7 checklist

- [ ] `kubelm-standard` (3B) trained and released
- [ ] `kubelm-pro` (7B) trained and released
- [ ] Full ladder benchmark published (edge + standard + pro)
- [ ] Helm chart updated for tier selection
- [ ] Blog post on the ladder and tradeoffs

---

## Ongoing throughout all phases

These are not phases — they happen continuously.

- **K8sGPT version tracking.** When K8sGPT ships a new version with MCP
  surface changes, evaluate impact. If material, plan a versioned kubelm
  release that pins to the new K8sGPT.
- **Public shipping cadence.** Something visible at regular intervals,
  even if small.
- **Community engagement.** Participate in K8sGPT issues and discussions.
- **Conference submissions.** KubeCon / KCD CFPs. Methodology talks count.

---

## What's NOT on the roadmap

- A web dashboard / UI (Helm chart + CLI is enough)
- Multi-language support (English only)
- Continuous learning / online updates (static releases only)
- Custom tool surfaces beyond K8sGPT's MCP server
- Snapshot-diagnosis prose generation (a different project; not this one)
- Replacing or competing with K8sGPT itself
- Additional infrastructure beyond GitHub, Hugging Face, and blog posts.
  No Discord, no separate website, no newsletter.

Adding any of these would expand scope. Don't.
