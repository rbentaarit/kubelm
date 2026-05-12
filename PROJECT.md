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
