# PROJECT.md

Internal planning document. The user-facing pitch lives in `README.md` — this
file captures the *thinking behind the project*: the thesis, why decisions
were made, what was considered and rejected, and what the architectural
commitments are.

This is the document any contributor (human or AI) should read first to
understand the project's intent, not just its current state.

---

## Core thesis

A small (1B–7B parameter) language model, fine-tuned on K8sGPT MCP tool-use
trajectories, can match the tool-use reliability of much larger generic
local models (Llama 3.3 70B, Qwen 2.5 32B) on K8sGPT's MCP surface, while
running on a single CPU node with no GPU.

The differentiator is **reliability on a specific tool surface**, not raw
intelligence. Generic small models routinely hallucinate tool names,
malform arguments, fabricate cluster state, and fail to terminate.
Specialization on the actual K8sGPT MCP tool schemas, argument formats, and
typical investigation patterns can close those failure modes — and that is
what makes a 3B model usable for production K8sGPT deployments where a 3B
model otherwise wouldn't be trusted.

## Why this thesis is plausible

- Tool-use behavior responds well to fine-tuning. Tool-call format and
  argument schemas are highly structured signals — exactly the kind of
  pattern small models can learn densely.
- The K8sGPT MCP surface is finite and stable enough to specialize for. Its
  evolution is gradual, public, and version-pinnable.
- Investigation trajectories follow recognizable patterns ("look at events
  → look at pod logs → conclude"). These patterns generalize across
  scenarios and are learnable from a moderate trajectory dataset.
- The K8sGPT team explicitly recommends local backends for production
  environments. That recommendation exists; the high-quality small local
  model fulfilling it does not.

## Why this thesis might be wrong

- Generic small models may already be reliable enough at K8sGPT MCP tool-use
  with good prompting. The Phase 3 baseline benchmark exists to find out.
- A curated prompt template + few-shot examples + a strong base model might
  close the gap without fine-tuning. If so, the answer is "ship the prompt
  template," not the model. Still a valuable artifact; project changes
  shape, doesn't end.
- The K8sGPT MCP surface may evolve faster than kubelm can track. If
  K8sGPT ships major tool-surface changes every few months, the maintenance
  burden of keeping kubelm pinned and re-trained may be untenable.
- Tool-use specialization may not transfer across base-model families. A
  technique that works for Llama 3.2 3B may not work for Qwen 2.5 3B. The
  ladder strategy depends on this transferring; if it doesn't, the project
  becomes single-base-model.

The benchmark phase exists specifically to surface these uncertainties before
significant effort is invested in fine-tuning.

---

## What this project is NOT

These are deliberate non-goals. Resist scope creep toward any of them.

- **Not a fork of K8sGPT.** Integrate via existing local-backend interfaces
  (`customrest`, `ollama`, `localai`).
- **Not a custom tool surface.** Track K8sGPT's MCP surface as canonical.
  No parallel tool ecosystem, no extending the surface, no diverging from
  what K8sGPT ships.
- **Not a remediation engine.** K8sGPT's auto-remediation architecture
  (Mutation CRs, operator policy gates, thresholds, rollback) handles
  destructive actions. kubelm proposes; the operator gates and disposes.
- **Not a frontier-model replacement.** For users with no
  cost/latency/hardware constraints, GPT-4-class cloud models will likely
  remain higher quality on the long tail.
- **Not a snapshot-diagnosis model.** That was an earlier framing
  considered and rejected. kubelm is for tool-use trajectories, not
  prose generation from frozen cluster state.
- **Not a research project.** Goal is shippable infrastructure using
  well-established techniques (QLoRA on small open base models, supervised
  fine-tuning on trajectory data). Innovation is in dataset construction,
  evaluation, and deployment story — not training algorithm.

---

## What "reliable tool-use" actually means

The benchmark measures specific failure modes. These are not aggregate
"accuracy" — each is tracked independently, because they fail in different
ways and matter to different users.

### Primary metrics

- **Tool-name hallucination rate.** Calls to nonexistent tools or misspelled
  names. Should approach zero on a model that has actually learned the
  surface.
- **Argument hallucination rate.** Arguments that don't match the tool's
  JSON schema (wrong types, invented fields, missing required fields).
- **Grounding failure rate.** Conclusions that reference cluster state never
  returned by any tool call. The model invents facts from its parametric
  knowledge instead of using tool results.
- **Termination failure rate.** Trajectories that loop, fail to reach a
  conclusion, or terminate prematurely without sufficient evidence.
- **Task completion rate.** End-to-end: did the trajectory reach a correct
  conclusion for the seeded scenario? (Subject to a defined rubric per
  scenario.)

### Secondary metrics

- **Steps-to-completion.** How efficient is the trajectory?
- **Per-step latency on target hardware.** Performance, measured on the same
  CPU spec as the deployment tier.
- **Token efficiency.** Total tokens generated across the trajectory.

A specialized small model that matches large-model task-completion rate
*and* has materially lower hallucination rates is the target outcome. That
combination justifies the project. Either alone does not.

---

## Methodology commitments

These are non-negotiable. Drift from them and the project loses its
credibility.

### 1. Eval-first

The benchmark is built before any fine-tuning. Every dataset addition,
every training run, every model release is measured against the same eval
suite. No vibe-based progress claims.

### 2. Real K8sGPT MCP server, no mocks

The eval harness boots a real K8sGPT MCP server (against a kind cluster
with seeded scenarios) and runs the model as an MCP client. Results
reflect deployed behavior, including tool-call schema validation,
session/protocol behavior, and the actual responses K8sGPT produces.

### 3. Version-pinned to K8sGPT

Every model release pins to a specific K8sGPT version. Training data,
eval scenarios, and benchmark results all reference the same K8sGPT
version. When K8sGPT evolves, kubelm follows in versioned releases.

### 4. Trajectory-based training data

Training examples are multi-step interactions
(goal → tool call → result → next tool call → ... → conclusion), not
diagnostic prose. The data structure matches the deployment behavior.

### 5. Reproducible training

Datasets, training scripts, hyperparameters, and seeds are public.
Anyone can re-run the training and verify the results.

### 6. Open-weight releases

All released models are Apache 2.0 with weights on Hugging Face. No
closed weights, no custom licenses, no gated downloads.

### 7. Public shipping cadence

Ship something publicly on a regular cadence. Without consistent public
shipping, work compounds privately and contributors can't engage.

---

## Architectural decisions

### Tool-use specialization, not diagnosis

This project is for tool-use behavior on K8sGPT's MCP server, not for
generating diagnostic prose from static cluster snapshots. The training
data is trajectories. The eval is multi-step. The metrics are tool-use
reliability metrics.

A snapshot-diagnosis model would be a different project. It was considered
and rejected because (a) K8sGPT's architecture is moving toward tool-use
via MCP, (b) the diagnosis-prose niche is contested by larger generic
models that have an inherent advantage, and (c) the unique gap is small
models that fail at tool-use specifically — not small models that fail at
prose generation.

### The model ladder

Three tiers, same training methodology, different base model sizes. Static
tier selection at install time.

| Tier              | Size      | Hardware              | Per-step latency  | Use case            |
|-------------------|-----------|-----------------------|-------------------|---------------------|
| `kubelm-edge`     | 1–1.5B    | 2-core CPU, 2GB RAM   | < 5 sec           | Edge, dev, CI       |
| `kubelm-standard` | 3B        | 4-core CPU, 4GB RAM   | 10–20 sec         | Production default  |
| `kubelm-pro`      | 7–8B      | 8-core CPU            | 15–30 sec         | Large/regulated     |

End-to-end trajectory time is per-step latency × number of steps. A typical
investigation runs 3–8 steps, so a `kubelm-standard` trajectory completes
in roughly 30 seconds to 2 minutes.

**Critical:** ship `kubelm-standard` first. Don't try to launch with the full
ladder. Prove the methodology works on the centerpiece, then expand.

### Performance is a primary design constraint

CPU performance for tool-use is determined by:

1. Model size and architecture (GQA/MQA, modern designs).
2. Quantization and inference engine (Q4_K_M baseline; CPU-optimized
   engines like llama.cpp / MLX).
3. Output structure (JSON tool calls are short by design — this works
   in our favor compared to prose-generation use cases).
4. Context management (tool results accumulate over a trajectory; pruning
   strategy matters for multi-step inference).

Performance is measured in the eval harness, not just reliability metrics.
A 2% reliability gain that doubles latency is rejected.

### Safety: architectural, not behavioral

K8sGPT's architecture handles destructive operations through Mutation
Custom Resources, operator policy gates, configurable risk thresholds, and
rollback mechanisms. The *system* gates destructive actions.

kubelm trains for *reliability properties* (correct tool calls, faithful
grounding, appropriate termination, structured output), not *safety
behaviors* (refusing dangerous tools, asking for confirmation, etc.).

This is intentional. Heavy refusal-pattern training would (a) duplicate
K8sGPT's existing safety architecture, (b) introduce false negatives where
the model refuses legitimate operations, (c) require subjective judgments
that don't generalize. The operator's policy layer is the right place for
those decisions.

The model's job: faithful tool-use. The operator's job: gating destructive
operations.

### Composite system, not just a model

The deployed system is: K8sGPT MCP server (cluster operations) → kubelm
(specialized tool-use) → K8sGPT operator and policy (safety, gating,
remediation execution).

Each component does one thing well. kubelm does not duplicate the others.

---

## Out-of-scope concerns deferred to later phases

Real concerns, but addressing them now would slow Phase 1–3 work.

- **Multilingual support.** English-only for v1.
- **Custom MCP servers beyond K8sGPT's.** v1 specializes for K8sGPT's MCP
  surface. Other K8s-related MCP servers (kubectl, helm, etc.) are not in
  scope. Revisit if user demand emerges and the K8sGPT MCP surface
  stabilizes.
- **Continuous learning / online fine-tuning.** Static model releases only.
- **Multi-cluster awareness.** v1 operates against single-cluster K8sGPT
  servers.
- **Agent-to-agent (A2A) protocols.** Not in scope. kubelm is an MCP
  client to K8sGPT's MCP server. Other agent communication patterns are
  out of scope.
- **Security audit / formal threat model.** Once the project sees real
  adoption, a proper supply-chain and adversarial-input threat model is
  needed. Not before.

---

## Decisions log

Append-only log of significant decisions. Update when major direction changes.

- **2026-05-05:** Project established. Repo created at
  github.com/rbentaarit/kubelm. README, LICENSE (Apache 2.0), PROJECT.md,
  ROADMAP.md, and CLAUDE.md committed as the foundation.
- **2026-05-05:** Name decision: `kubelm` over `kubellm`. Rationale: easier
  to type, matches K8s naming convention (kubelet, kubeadm, kubectl), and
  doesn't overclaim "Large" for what are actually small models.
- **2026-05-05:** Project framing locked: tool-use specialist for K8sGPT
  MCP, not snapshot diagnosis. The diagnosis framing was considered and
  rejected after reviewing K8sGPT's roadmap (MCP server, auto-remediation
  architecture). The unique gap kubelm fills is reliability on K8sGPT MCP
  tools at small CPU-only sizes — not diagnostic prose generation.
- **2026-05-05:** Scope locked to K8sGPT's MCP surface as canonical. No
  custom tool surface, no parallel ecosystem. Versioned to track K8sGPT
  releases.
- **2026-05-05:** Safety model: architectural (K8sGPT operator + Mutation
  CRs + policy gates), not behavioral (model refusal patterns). The model
  is trained for tool-use reliability; the operator is responsible for
  destructive-operation gating.
- **2026-05-05:** Tier order: ship `kubelm-standard` (3B) first, then edge
  and pro variants. Single-model launch reduces v1 scope.
- **2026-05-11:** Cloud frontier model for the Phase 3 baseline locked to
  `gpt-5.4`. Newer OpenAI models (`gpt-5`, `gpt-5.5`) reject
  `temperature: 0` ("Only the default (1) value is supported") and would
  force stochastic sampling — breaking methodology commitment #5
  (reproducible eval). `gpt-5.4` is the latest model that is both
  current and deterministic. Backend code routes `max_tokens` →
  `max_completion_tokens` automatically for the gpt-5 family and
  o1/o3 reasoning models (`eval/runner/openai_backend.py`). Re-check
  against the live `/v1/models` catalog at each baseline refresh —
  the OpenAI family moves fast and the temp=0 constraint may shift.
- **2026-05-11:** K8sGPT MCP `list-resources` tool (v0.4.32) does not
  expose `networkpolicies` as a supported resourceType. Surfaced from a
  drill-in on the `network-policy-block-001` scenario: gpt-5.4 correctly
  called `list-resources(resourceType=networkpolicies)`, K8sGPT
  responded with `isError: true / unsupported resource type`, and the
  scenario's rubric (which requires the specific policy name
  `default-deny-ingress`) became unsatisfiable on the current MCP
  surface. Documented in `eval/results/summaries/README.md`. Two
  follow-ups: file upstream against K8sGPT, and either relax the
  scenario's rubric or skip the scenario until upstream lands.
- **2026-05-11:** `ref_pass` metric in `eval/metrics/reference_calls.py`
  fixed to gate `must_include` / `any_of` matches on the corresponding
  `tool_result.is_error != true`. Previously any call that matched a
  matcher's name/args counted as a reference call even when the MCP
  server had rejected it — over-stating the column. `forbidden`
  matchers still fire on errored calls (the attempt is the violation,
  not the result). The 2026-05-11 Shape B summary was re-graded
  against the fix; three cells flipped True→False, all on
  `network-policy-block-001` (the K8sGPT-MCP unsupported-type case).
  See commit `ee8c75c` and the regrade helper
  `eval/results/summaries/regrade_ref_pass.py`.
- **2026-05-11:** Scenario library cleared the ROADMAP Phase 2 lower
  bound of 30 (was 10 at start of session). Added scenarios for
  readiness/liveness probes, init containers, ConfigMap-missing,
  ServiceAccount-missing, taint-no-toleration, anti-affinity,
  insufficient-CPU, pod-PVC-not-found, StatefulSet+PVC,
  Deployment-rollout-stuck, Deployment-paused, CronJob-suspended,
  Job-backoff-exhausted, DaemonSet-no-fit, HPA-no-metrics,
  Service-port-mismatch, ResourceQuota-CPU-exceeded,
  init-image-pull, pod-command-not-found. Authoring patterns
  captured in `CLAUDE.md`.
- **2026-05-12:** Third Shape B baseline published
  (`shape-b-2026-05-12.json`) — 5 models × 30 scenarios. First cut
  against the expanded library. The findings strengthen the
  hypothesis that motivates this project:

  1. The 3B → 7B phase change is fully robust at n=30: llama3.2-3b
     hits 1/30 complete, 6/30 rubric, 0/30 ref_pass. The capability
     cliff for tool-use is real and sharp.
  2. The 7B–32B–gpt-4o plateau on rubric is preserved (24-26/30),
     confirming "more parameters past 7B doesn't buy more rubric on
     this surface" with 3x the sample.
  3. **gpt-5.4 has 30/30 grounding failure** — every conclusion
     contains at least one claim not derivable from any tool
     result, while rubric is 29/30. The frontier model is reliably
     producing the right answer wrapped in fabricated supporting
     detail. This is the strongest direct evidence so far that the
     "small + grounded > frontier + verbose" thesis has room to
     win on a metric that matters for production. Worth a
     per-scenario audit before publishing externally (the
     grounding analyzer is rule-based and a shared formatting tic
     could partially explain the score), but the gap from
     qwen2.5-7b (14/30) and gpt-4o (12/30) is too large to
     dismiss.
  4. **qwen2.5-7b is the candidate base model that a kubelm
     fine-tune would have to beat.** 30/30 complete, 29/30
     ref_pass, 24/30 rubric, 14 grounding failures. Competitive
     with gpt-4o on grounding (14 vs 12) at 4.7 GB. If kubelm
     can match qwen2.5-7b on rubric AND ground better, the
     thesis is validated; if it merely matches all metrics, the
     decision gate after Phase 3 has to weigh "specialized but
     no clear win" against "ship a curated prompt template
     instead".
- **2026-05-12:** Per-scenario audit of gpt-5.4's 30/30 grounding
  failure overturns the prior interpretation. The headline reading
  was "frontier model reaches the right answer wrapped in
  fabricated supporting detail." Walking the ungrounded-fact list
  across all 30 scenarios shows that the vast majority of flagged
  facts are actually present in tool output — but rendered by
  gpt-5.4 in formats the rule-based grounding analyzer can't
  match:

  - YAML-path notation: `configMapKeyRef.name: app-settings`
    (analyzer expects a substring match against raw JSON like
    `{"configMapKeyRef":{"name":"app-settings"}}` and fails)
  - Dotted status paths: `state.waiting.reason: CrashLoopBackOff`
    rendered from a JSON object whose path the model traversed
  - Quoted-vs-unquoted: `targetPort: "http-port"` vs the
    `targetPort: http-port` actually in the spec
  - Reasonable inferences with synthesized strings:
    `http://<pod>:80/healthz` composed from a probe spec, not
    literally present in any single tool result
  - Scenario-context fills for unsupported K8sGPT MCP types
    (e.g., NetworkPolicy names — see prior decisions log entries)

  Genuine fabrications across 30 scenarios are roughly a handful.
  The dominant pattern is gpt-5.4 producing a more structured,
  YAML-shaped, faithful representation of tool output than the
  raw text it came from, which the analyzer's substring matcher
  treats as ungrounded.

  Implications for the project:

  1. **The "frontier hallucinates supporting detail" narrative is
     retracted** in its current form. The audit doesn't show
     gpt-5.4 inventing facts; it shows a structured paraphrase
     that the v1 grounding analyzer can't follow. The blog draft
     and the prior 2026-05-12 decisions-log entry are revised
     to reflect this.
  2. **The grounding metric needs a v2** that tolerates structural
     rephrasing (dotted paths, quote variants, YAML notation,
     reasonable string composition from primitives). Without
     this, cross-model grounding comparisons aren't reliable —
     verbose models will systematically lose, terse models will
     systematically win, and neither result reflects faithfulness.
  3. **The kubelm thesis is not disproved, but the evidence for
     it from this metric is weakened.** The qwen2.5-7b "candidate
     base model" framing still holds on rubric and ref_pass; the
     grounding-as-distinguishing-axis argument has to wait for
     metric v2 before it can carry weight.
  4. **Methodology principle reinforced:** rule-based metrics
     need adversarial audit against verbose-but-faithful models
     before being trusted at face value, especially when the
     bench reports a striking signal. We caught this before
     publishing — the 2026-05-12 caveat about per-scenario
     audit was load-bearing.
- **2026-05-12:** Phase 4 prep scaffolding committed. The trajectory
  training-data format (`data/seed/FORMAT.md`, schema_version 1) is
  pinned around an OpenAI-shaped `messages` array so existing SFT
  toolchains (HF TRL, Axolotl, Unsloth) can ingest natively. The
  format separates `messages` (training payload) from `provenance`
  and `quality` (metadata + eval-harness read-outs) so downstream
  loops can `if "messages" in record:` without dragging metadata
  into the model input. A `data/seed/convert.py` script
  back-converts existing eval results into this format; the first
  seed file (`data/seed/v0/gpt-5.4-2026-05-12.jsonl`) is the 29
  rubric-passing gpt-5.4 trajectories from the 2026-05-12 Shape B
  cut. The `tools` field is currently `null` in this seed because
  the eval harness doesn't persist the K8sGPT MCP `tools/list`
  payload alongside trajectories; a `data/seed/snapshot_tools.py`
  helper exists to capture them per K8sGPT version (run on demand),
  and a `TrajectoryRecorder` change to persist tools at run time is
  on the ROADMAP followup list. Phase 4 step 1 ("generate seed
  trajectories using a strong model + manual review") is reframed
  as a *reuse* of the gpt-5.4 bench output rather than a separate
  generation pass — the strong model already ran 30 scenarios with
  29/30 rubric-pass, so converting + reviewing those seeds is
  Phase 4's actual cheapest path to v0.1.
- **2026-05-13 (revised same day):** Phase 5 v0 target retargeted
  from `kubelm-standard` (3B) to `kubelm-edge` (1.5B). The original
  decision earlier this day landed on `Qwen/Qwen2.5-3B-Instruct` as
  the kubelm-standard base. On reflection that was the wrong tier
  to ship first: the project's deployment story is "K8sGPT
  alongside a small model in a standalone or dev cluster", which
  argues for the edge tier (1-1.5B per the ROADMAP model ladder)
  as v0.

  Retargeted base: `Qwen/Qwen2.5-1.5B-Instruct`. Inputs to the
  decision:

  1. **Same-day baseline measurement.** Ran `qwen2.5:1.5b` against
     the 30-scenario library (1 model × 30 scenarios, ~17 min
     wall-time, results in
     `eval/results/summaries/shape-b-2026-05-13-qwen-1.5b.json`):

       - complete:         8/30
       - schema_passed:    27/30
       - name_halluc:      0
       - arg_halluc:       2
       - grounding_failed: 16/30
       - ref_pass:         3/30
       - rubric_pass:      10/30
       - errored:          1 (settle-race on pod-anti-affinity)

     The model is not catastrophically broken (unlike `llama3.2:3b`
     at 1/30 complete, 6/30 rubric, 0/30 ref_pass). 8/30 complete +
     10/30 rubric out of the box is a real foothold for SFT.

  2. **HF survey for K8s-specialized small models came up empty
     for this surface.** Queried `kubernetes / kubectl / k8s /
     devops / sre` filtered to text-generation. At <2B params,
     the candidates were:

       - chowmean/k8s_Qwen2.5-0.5B-Instruct (Qwen 2.5 0.5B base, 4
         downloads, trained on `kubernetes_commands` — command Q&A,
         not multi-step trajectory)
       - lakhera2023/devops-slm (Qwen 2 0.5B base, 29 downloads,
         broad devops/docker/cicd, older Qwen 2 not 2.5)
       - brito-parzivall/tinyllama-kubectl-v3 (TinyLlama 1.1B, 3
         downloads, stale since 2024-03)

     All have smaller bases than 1.5B and are trained for different
     surfaces (command Q&A, kubectl help, generic devops chat).
     Our 365-trajectory K8sGPT-MCP-specific corpus is more
     on-target than what any of those models were trained on, and
     starting from a larger / better-tuned base (Qwen 2.5 1.5B
     Instruct) means the SFT lift is more achievable. The
     family-control argument (same family as the 7B empirical
     target) wins.

  3. **Deployment footprint matches the dev-cluster framing.**
     Q4_K_M GGUF: ~1.0 GB on disk. Working set at 8K context:
     ~1.5–1.7 GB (weights + KV cache + compute). K8s Pod
     resources: requests 1 cpu / 1.5Gi memory, limits 2 cpu / 2Gi
     — consistent with the ROADMAP `kubelm-edge` tier definition
     ("2-core CPU, 2GB RAM"). Per-step latency on 2-4 cores:
     ~15-50 tokens/sec → typical 2-5 turn investigation runs in
     30s-3min.

  Training data: positives only for v0 (29 seeds + 290 variants =
  319 trajectories). Synthetic negatives excluded because all 46
  carry review_status: unreviewed and the recovery prose is
  templated.

  Framework: Unsloth (QLoRA 4-bit). Cost envelope per training
  run is now ~$1.50-$6 on a rented A100 (smaller model, faster
  per-step), down from the $10 envelope of the 3B target. 2-4
  hours of A100 time per run.

  Scaffolding committed: `training/configs/kubelm-edge-v0.yaml`
  (renamed from kubelm-standard-v0.yaml; LoRA r=32 unchanged but
  batch sizing tweaked for the smaller model: per_device 8 vs 4,
  grad_accum 2 vs 4 → same effective batch 16),
  `training/sft.py`, `training/eval_checkpoint.py`,
  `training/README.md`. The 1.5B baseline summary
  (`eval/results/summaries/shape-b-2026-05-13-qwen-1.5b.json`)
  joins the published cuts.

  Quality bar for release (revised against the new baseline):

    - Minimum: rubric ≥ 12, complete ≥ 12, ref_pass ≥ 6, 0 name
      hallucinations, ≤ 2 arg hallucinations
    - Stretch: rubric ≥ 17, complete ≥ 20, ref_pass ≥ 12
    - Optimistic: match qwen2.5:7b (rubric 24, complete 30,
      ref_pass 29) — "specialization fully recovered the
      capability lost going 7B→1.5B in the same family"

  After kubelm-edge ships, Phase 7's ladder expansion produces
  kubelm-standard (3B) and kubelm-pro (7B) on the same dataset
  and recipe. Edge first matches the deployment story; the
  larger tiers follow for clusters with more memory.
- **2026-05-12:** 70B GPU-box benchmark dropped from Phase 3.
  Originally planned to confirm the "above 7B is flat" finding at
  the largest open-weight tier; the 2026-05-12 cut showed
  qwen2.5:32b (rubric 26/30, ground_fail 12) ≈ gpt-4o (rubric 25/30,
  ground_fail 12) ≈ qwen2.5:7b (rubric 24/30, ground_fail 14),
  which is sufficient evidence that the plateau extends from 7B
  to the frontier. A 70B point would only confirm what's already
  visible and would consume external compute the project doesn't
  need at this gate. If a later phase revisits the local-large
  tier (e.g. for `kubelm-pro`'s 7-8B baseline comparison), the
  GPU-box infra can be reintroduced then.
- **2026-05-14:** **kubelm-edge v0 locked to attempt-2 (2 epochs).**
  Two QLoRA training attempts on an A100 SXM4-80GB against the v0
  corpus (319 trajectories = 29 seeds + 290 mechanical variants,
  positives only). Both committed to `eval/results/summaries/`:
  `kubelm-edge-v0-attempt-1-2026-05-14.json` and the v0 release
  `kubelm-edge-v0-2026-05-14.json`.

    attempt-1 (3 epochs, train_loss 0.27, plateau bottom 0.01):
      complete 21/30 | rubric 17/30 | ref_pass 19/30
      ground_fail 21 | arg_halluc 2 | name_halluc 0 | errored 1

    attempt-2 (2 epochs, train_loss 0.42, plateau bottom 0.07):
      complete 29/30 | rubric 23/30 | ref_pass 21/30
      ground_fail 27 | arg_halluc 0 | name_halluc 0 | errored 1

    base qwen2.5:1.5b (2026-05-13 baseline, for delta context):
      complete 8/30  | rubric 10/30 | ref_pass 3/30
      ground_fail 16 | arg_halluc 2 | name_halluc 0 | errored 1

    qwen2.5:7b (2026-05-12 Shape B, the 4-5× empirical target):
      complete 30/30 | rubric 24/30 | ref_pass 29/30
      ground_fail 14 | arg_halluc 0 | name_halluc 0 | errored 0

  Findings:

  1. **The 3-epoch run overtrained.** attempt-1's training loss
     bottomed at ~0.01 by mid-epoch 2 and stayed there — classic
     memorization signal for 319 examples × 36.9M trainable LoRA
     params. attempt-2's cosine schedule re-cast over 2 epochs
     (instead of 3) stopped at the plateau-start zone (loss ~0.07)
     before the deep memorization tail. Result: +6 rubric,
     +8 complete, -2 arg_halluc.

  2. **attempt-2 essentially matches qwen2.5:7b on the two
     headline metrics** at 1/4-to-1/5 the deployment footprint:
     - rubric: 23 vs 24 (1 short)
     - complete: 29 vs 30 (1 short; the 1 we lose is the
       pod-anti-affinity-001 settle-race that also errored on
       attempt-1 and the published 2026-05-13 baseline — a
       scenario harness issue, not a model issue)
     - ref_pass: 21 vs 29 (8 short — still the largest gap)
     This is the strongest direct evidence to date that a small
     specialist can hold its own against a 4-5× larger general
     model on the K8sGPT MCP surface, which is the project's
     core thesis.

  3. **Grounding regressed in both attempts** vs base
     (16 → 21 → 27). Both attempts are worse than the base 1.5B
     here, which makes the metric the prime suspect, not the
     fine-tunes. Same pattern that prompted the 2026-05-12 audit
     of gpt-5.4's "frontier hallucinates" reading: the rule-based
     grounding analyzer doesn't tolerate structural rephrasing
     (YAML-shaped output, dotted paths, quoted-vs-unquoted), and
     fine-tuning is exactly the operation that shifts the model's
     output style. Per-scenario audit of attempt-2's 27 flagged
     facts is the right next step before publishing the grounding
     number externally; not a release blocker.

  4. **Quality bars cleared:**
       Min:        rubric ≥ 12 ✓ (23) | complete ≥ 12 ✓ (29)
                   ref_pass ≥ 6 ✓ (21) | name_halluc 0 ✓ | arg_halluc ≤ 2 ✓ (0)
       Stretch:    rubric ≥ 17 ✓ (23) | complete ≥ 20 ✓ (29)
                   ref_pass ≥ 12 ✓ (21)
       Optimistic: rubric ≥ 24 ✗ (23) | complete ≥ 30 ✗ (29)
                   ref_pass ≥ 29 ✗ (21)

     attempt-2 clears the release bar and stretch bar in full;
     misses optimistic by 1 on rubric, 1 on complete (the harness
     settle-race), and 8 on ref_pass.

  Cost envelope per attempt was ~$0.20-1.00 on a rented A100;
  total Phase 5 GPU spend across both attempts ~$2. Both attempts
  were 1.5B QLoRA r=32 alpha=64 with the same data and the same
  Unsloth + trl 0.24 pinning; only `num_train_epochs` differed
  (3 vs 2). attempt-2's 2-epoch retrain is the v0 release.

  Lessons captured in `training/runpod_setup.sh` and
  `training/runpod_finalize.sh` so attempt-3+ won't re-litigate
  the same launch-day gotchas: torch=system pin, nvidia-cu12-*
  skip list, .venv on local NVMe instead of MFS, llama.cpp build
  on local NVMe, Python interpreter probed for torch (not blind
  `python3`).

- **2026-05-19:** v0.1 plan agreed. Six-stage arc to close the
  hallucination measurement gap before v0.1 model training:
  Stage 1 (narrative-consistency metric), Stage 2 (grounding audit),
  Stage 3 (grounding metric v2 + retroactive re-grade),
  Stage 4 (rubric tightening + adversarial scenarios),
  Stage 5 (re-run baselines under the new metrics),
  Stage 6 (v0.1 model train). Stages 1-3 shipped same-day at commits
  `bab8733`, `a9218be`, `776b085`, `788912c`, `5ff0c2b`, `3184dee`.

- **2026-05-19:** Stage 3 grounding metric v2 retracts the strict
  reading of v0's `ground_fail: 27/30` from the 2026-05-14 ship memo.
  The v1 analyzer was rule-based and intolerant of structural
  rephrasing (JSON ↔ dotted notation, CamelCase ↔ hyphenated,
  quoted ↔ unquoted). Stage 2's manual audit of all 114 v0 ungrounded
  facts labeled them under a 5-category taxonomy and found
  **12.3% (14 of 114) are real fabrications**; 87.7% are metric
  blind-spots (62.3% structural_rephrase, 25.4% composed_inference,
  0% scenario_fill, 0% unsupported_tool). Stage 3 built `grounding_v2`
  calibrated against those labels (fab P=91.7% / R=85.7%, rephrase
  P=100% on the full set; P=88.9% / R=85.2% / 100% under
  leave-one-scenario-out cross-validation). The retroactive re-grade
  (commit `3184dee`) inverts the v0-vs-base story:

      base qwen2.5-1.5b   v1 ground_fail 16/29  →  v2 fab_runs 14/29  fabs 43
      attempt-1           v1 ground_fail 21/29  →  v2 fab_runs  6/29  fabs  9
      v0 (attempt-2)      v1 ground_fail 27/29  →  v2 fab_runs  9/29  fabs 13

  The fine-tunes have **3-5× fewer fabrications than the base 1.5B
  model**, not more. The v1 increase (16 → 21 → 27) was almost
  entirely style drift — fine-tuning adopted JSON-faithful output
  that v1's normalization couldn't follow but v2's can.

  This also retracts the 2026-05-12 reading of gpt-5.4's "30/30
  grounding paradox" — under v2 the same trajectories score
  **3/30 fab_runs / 3 fabs**. The earlier audit caveat ("audit
  per-scenario before drawing") was vindicated by Stage 3 running
  that audit end-to-end via the new analyzer.

  v0's claim from the ship memo ("matches qwen2.5:7b at ~1/4 the
  deployment footprint") holds on the four shipped metrics; on the
  v2 grounding metric v0 is off by 8 fabrications vs qwen2.5:7b
  (13 vs 5 on the 30-scenario library). Real but small.

  Schema-version semantics: `RESULTS_SCHEMA_VERSION` 1 → 2 → 3 and
  `BENCH_SCHEMA_VERSION` 1 → 2 → 3 track the three metric
  generations. Schema 3 redefines `grounding_failed` from "any
  ungrounded fact" to "fabrication present"; the v1 boolean is
  preserved as `grounding_v1_report.has_grounding_failure` for
  backward comparison. All seven committed summaries are at
  schema 3 as of `3184dee`. The HF model card grounding caveat
  for `kubelm-edge-v0` should be softened to reflect the v2
  reading; that update is deferred to a small standalone commit
  before Stage 4 finalizes.

- **2026-05-20:** v0.1 Stage 6 first training iteration — the
  **data-diversity lever** — is a recorded **negative result**. It
  did NOT close the `ref_pass` gap it targeted (Stage 5 showed v0 at
  ref_pass 24/33 vs qwen2.5:7b's 32/33; everything else at parity).
  The iteration rebuilt the corpus with two generator styles
  (gpt-5.4 + qwen2.5-7b, the reference target, ref_pass 32/33) across
  the 33-scenario library, v2-fabrication-filtered (550 records vs
  v0's 319), recipe held constant (QLoRA r=32, 2 epochs, lr 2e-4).
  Trained on RunPod RTX 6000 Ada, evaluated locally. Result
  (`eval/results/summaries/kubelm-edge-v01-2026-05-20.json`):

      metric      base 1.5B   v0    v0.1   qwen2.5-7b
      ref_pass      4/33     24/33  23/33    32/33
      rubric       12/33     24/33  21/33    25/33
      fabs           52        7      2        6
      complete     10/33     30/33  32/33    32/33

  v0.1 vs v0: ref_pass flat (23 vs 24), rubric regressed (21 vs 24),
  but grounding improved sharply (fabs 2 vs 7 — now frontier-level)
  and complete +2. A lateral trade, not the targeted upgrade; clearly
  better than its base on every column but not better than v0 on the
  metric that mattered.

  Two conclusions: (1) **ref_pass looks like a 1.5B capacity ceiling,
  not a data-coverage problem** — training the student directly on
  the reference target's high-ref_pass trajectories did not transfer
  reference-call discipline; the data-diversity hypothesis is
  disproven for this gap. (2) The rubric regression + a low per-step
  train_loss (~0.025 vs v0 attempt-2's ~0.07) indicates **overfitting**
  — 550 records at 2 epochs over-trained relative to v0's 319. The
  v2-fabrication filter is what delivered the grounding win (excluding
  generator-fabricating trajectories taught more faithful output).

  Per the eval-first / no-cherry-pick commitments, v0.1 is NOT
  released; it is recorded as-is. Next iteration pulls the **recipe
  lever** (1 epoch + a lr-1e-4 variant on the same corpus) to test
  whether killing the overfit recovers rubric and/or moves ref_pass.
  If recipe-alone can't move ref_pass, the remaining lever is a
  base bump to 3B (capacity), which changes the edge-tier deployment
  story and is a separate decision. Artifacts (GGUF + LoRA) kept
  locally at `~/kubelm-v01-artifacts/`, not published. RunPod
  training is now fully CLI-reproducible via runpodctl
  (`training/runpod_setup.sh` + `runpod_finalize.sh`, the latter
  fixed for PEP 668 in commit `66286fc`).

- **2026-05-21:** **reference_calls v2** — the "ref_pass capacity
  ceiling" from the 2026-05-20 sweep is largely RETRACTED; it was a
  metric artifact, like grounding v1→v2. Auditing v0.1's 9 ref_pass
  failures found ~5 of 9 were the `reference_calls` matchers rejecting
  *valid* investigation calls, not the model failing. The `any_of`
  specs were inconsistent: 9 of 33 scenarios (the oldest) omitted
  `analyze` from `any_of` though it's K8sGPT's canonical tool and the
  other 24 credit it; a few `list-events` matchers keyed on
  `namespace=` instead of `involvedObjectName=`, and a few
  `list-resources` matchers omitted the type the model listed — all
  violations of the scenario-authoring rules already in CLAUDE.md.
  v2 adds `analyze(namespace)` to all 33 + patches the three
  demonstrated gaps (metric code unchanged; specs only). Re-graded on
  existing trajectories (no retrain):

  (v2 landed in two passes: pass 1 added analyze; a completion pass
  added list-resources(pods/pod) to 6 pod-subject scenarios + a
  get-resource(pod) gap — more matcher gaps the v0 failure
  trajectories exposed.) Final:

      model            old ref_pass   v2 ref_pass
      qwen2.5-1.5b        4/32           4/32     (+0, genuine)
      kubelm-edge-v0     24/33          31/33     (+7)
      kubelm-edge-v0.1   23/32          28/32     (+5)
      qwen2.5-7b         32/32          32/32     (+0)
      gpt-5.4            33/33          33/33     (+0)

  **The correction is real, not inflation: the reference models
  (qwen2.5-7b, gpt-5.4) didn't move** — they already made the credited
  calls — while base 1.5b stayed at 4 (it genuinely doesn't
  investigate). Only the mid models that used valid-but-uncredited
  calls gained. So **v0's true ref_pass is 31/33 — at parity with
  qwen2.5-7b (32)**; its only 2 fails are network-policy-block (the
  documented K8sGPT can't-expose-networkpolicies tool-surface limit)
  and pvc-unbound (the one genuine model failure). The "ref_pass gap"
  was almost entirely a metric artifact. kubelm-edge-v0 stays the
  best-balanced model (ref_pass 31 + rubric 24). The committed
  Shape C + v0.1 digests are re-graded to v2. Lesson (third time now,
  after grounding and this): **audit whether a metric is measuring the
  model or its own strictness before concluding a capability limit.**
  Remaining genuine ref_pass failures (v0: just pvc-unbound; v0.1:
  configmap/liveness-oom/pending/service-selector — no investigation,
  wrong
  resource, identifier confusion) are the targeted-data candidates
  for the next iteration.
