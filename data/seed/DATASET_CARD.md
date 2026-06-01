---
license: cc-by-4.0
language:
- en
pretty_name: kubelm seed trajectories v0
size_categories:
- n<1K
tags:
- kubernetes
- k8sgpt
- mcp
- tool-use
- sft
- trajectories
---

# kubelm seed trajectories v0

A small, curated corpus of multi-step tool-use trajectories against
[K8sGPT](https://k8sgpt.ai)'s MCP server. Each trajectory is one
Kubernetes investigation: a goal statement, a sequence of K8sGPT
MCP tool calls + responses, and a final conclusion. The corpus is
designed for supervised fine-tuning of small local models that
need to use K8sGPT's MCP tools reliably on commodity CPU hardware.

This is **v0**: a foundational dataset, not a production training
corpus. It demonstrates the methodology and provides a starting
point for iteration. See "Intended use" and "Limitations" below.

## Quick facts

- **Total trajectories:** 365
- **Unique scenarios:** 29 (covers pod-startup, service/networking,
  scheduling, storage, RBAC, resources, and workload-controller
  failure modes — see [the scenario library](https://github.com/rbentaarit/kubelm/tree/main/eval/scenarios/specs))
- **K8sGPT version pin:** `0.4.32`
- **MCP protocol version:** `2025-03-26`
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Composition

| File | Records | Source | Review status |
|---|---|---|---|
| `v0/gpt-5.4-2026-05-12.jsonl` | 29 | `eval_bench` (gpt-5.4 trajectories) | 29/29 `accepted` |
| `varied/v0/gpt-5.4-2026-05-12-varied.jsonl` | 290 | `eval_bench_variation` (10× surface-detail substitutions per seed) | inherited `accepted` |
| `varied/v0/negatives-2026-05-12.jsonl` | 46 | `negative_synthetic` (2 injection patterns) | 46/46 `unreviewed` |

The 29 positive seeds come from running gpt-5.4 (OpenAI) against
the [kubelm Phase 2 scenario library](https://github.com/rbentaarit/kubelm/tree/main/eval/scenarios/specs)
through the [Phase 1 eval harness](https://github.com/rbentaarit/kubelm/tree/main/eval).
The 30 scenarios in that library were run on the 2026-05-12 Shape
B benchmark; one (pod-insufficient-cpu) had a rubric failure and
was excluded. Every included trajectory passes the eval harness's
`conclusion_rubric_passed` check.

The 290 variants are deterministic surface-detail substitutions of
the seeds. The substitution dimensions are the namespace name
(replaced from a 30-name pool of realistic K8s namespace strings)
and the primary resource names (replaced from role-specific pools).
Substitutions are consistent within a trajectory — every
occurrence across the system prompt, goal, every tool call, every
tool result, and the final conclusion uses the variant strings.
The point is to teach the model that the *structural pattern* of
each failure mode matters, not specific identifier strings.

The 46 negatives are synthetic and demonstrate two recovery
patterns:

- **wrong_resource_type** (17): the trajectory begins with a
  `list-resources` call using a plausible typo for `resourceType`
  (e.g., `pod_name` instead of `pods`). The K8sGPT MCP server
  responds with its standard "unsupported resource type" error.
  An assistant turn acknowledges the error and the trajectory
  continues with the correct call.
- **hallucinated_tool_name** (29): the trajectory begins with a
  call to `get-pod` — a plausible-looking but non-existent tool
  (the real tool is `get-resource`). MCP responds with
  `unknown tool: get-pod`. The assistant recovers, then proceeds
  with the intended investigation.

The error response strings in the negatives reproduce the actual
K8sGPT MCP error shape observed in the source corpus (see
[the 2026-05-12 audit](https://github.com/rbentaarit/kubelm/blob/main/eval/results/summaries/README.md)).
They are NOT invented.

## Record format (schema_version 1)

One JSON object per line. Every record is a complete training
example with no nested context to reconstruct. Top-level fields:

```json
{
  "schema_version": 1,
  "trajectory_id": "<uuid4>",
  "k8sgpt_version": "0.4.32",
  "mcp_protocol_version": "2025-03-26",
  "scenario_id": "configmap-missing-001",
  "scenario_source_path": "eval/scenarios/specs/configmap-missing-001.yaml",
  "provenance": { "source": "...", "generator_model": "...", ... },
  "system_prompt": "...",
  "goal": "...",
  "tools": null,
  "messages": [ ... ],
  "quality": { ... }
}
```

The `messages` array is in OpenAI tool-use format — directly
trainable by Hugging Face TRL, Axolotl, Unsloth, or any framework
that accepts `{role, content, tool_calls, tool_call_id}`:

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "", "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]},
  {"role": "tool", "tool_call_id": "...", "content": "..."},
  {"role": "assistant", "content": "<final conclusion>"}
]
```

`tool_calls.function.arguments` is a JSON-encoded string per
OpenAI's wire format. `tool` message content carries the
JSON-stringified MCP tool result (the wrapper is
`{"content":[{"type":"text","text":"..."}], "isError": bool}` —
inspect at training time to recover the inner text if needed).

Full schema with field-by-field commentary is in
[FORMAT.md](https://github.com/rbentaarit/kubelm/blob/main/data/seed/FORMAT.md).

## Provenance

Every record has a `provenance` object linking back to its source:

```json
{
  "source": "eval_bench" | "eval_bench_variation" | "negative_synthetic",
  "source_run_id": "...",            // eval bench run id
  "source_bench_id": "...",          // eval bench id
  "generator_model": "gpt-5.4",
  "generator_backend": "https://api.openai.com/v1",
  "generated_at": "2026-05-12T...",
  "license": "CC-BY-4.0",
  "review_status": "accepted" | "accepted_with_edits" | "unreviewed" | "rejected"
}
```

For variants, additional fields point at the source trajectory:

```json
{
  "variation_of": "<source trajectory_id>",
  "variation_idx": 0,
  "variation_map": { "scenario-foo-001": "prod-api", "api-pod": "auth-service", ... }
}
```

For negatives:

```json
{
  "negative_of": "<source trajectory_id>",
  "negative_pattern": "wrong_resource_type" | "hallucinated_tool_name"
}
```

## Quality block

Each record carries the eval harness's read-outs:

| field | meaning |
|---|---|
| `termination_label` | `complete`, `premature`, `errored`, `truncated` |
| `schema_passed` | All tool calls passed schema validation |
| `schema_name_halluc` | Count of hallucinated tool names (should be 0) |
| `schema_arg_halluc` | Count of malformed-argument tool calls |
| `reference_calls_passed` | Trajectory hit the scenario's reference-call expectations |
| `conclusion_rubric_passed` | The single most important quality signal |
| `grounding_failed` | v1 grounding analyzer flagged at least one claim |
| `grounding_failed_v1_artifact` | Reviewer's verdict on whether the v1 flag is a formatting artifact (see Limitations below) |
| `step_count` | Number of assistant turns in the trajectory |
| `model_latency_ms` | Sum of model latency across all assistant calls |

## Intended use

- Supervised fine-tuning of small (1–7B) local models for
  K8sGPT MCP tool-use, especially on commodity CPU hardware.
- Bootstrapping a baseline before constructing a larger corpus.
- Reference / benchmark dataset for the kubelm project's
  Phase 5 (first fine-tuned model) and Phase 7 (model ladder
  expansion).

### Models trained on this dataset

- **`kubelm-qwen2.5-1.5b-v1`** (Qwen2.5-1.5B base; formerly
  `kubelm-edge-v0`) — released 2026-05-14. Trained on the 319 positives
  (29 seeds + 290 variants); negatives excluded because all 46 carry
  `review_status: unreviewed`.
  - GGUF: [`rbentaarit/kubelm-qwen2.5-1.5b-v1`](https://huggingface.co/rbentaarit/kubelm-qwen2.5-1.5b-v1) · LoRA: [`…-1.5b-v1-lora`](https://huggingface.co/rbentaarit/kubelm-qwen2.5-1.5b-v1-lora)
  - Eval result vs base 1.5B: `complete` 8→29/30, `rubric_pass`
    10→23/30, `ref_pass` 3→21/30. Full row in
    [`eval/results/summaries/kubelm-edge-v0-2026-05-14.json`](https://github.com/rbentaarit/kubelm/blob/main/eval/results/summaries/kubelm-edge-v0-2026-05-14.json).
- **`kubelm-qwen3.5-2b-v1`** (Qwen3.5-2B base; formerly
  `kubelm-edge-v0.3`) — released 2026-05-27. Trained on the v0.2 corpus
  (561 records, see below). The headline deployable.
  - GGUF: [`rbentaarit/kubelm-qwen3.5-2b-v1`](https://huggingface.co/rbentaarit/kubelm-qwen3.5-2b-v1) · LoRA: [`…-2b-v1-lora`](https://huggingface.co/rbentaarit/kubelm-qwen3.5-2b-v1-lora)
  - Eval result on the 35-scenario library: `rubric_pass` 32/35,
    `ref_pass` 32/35, `fabrications` 3, `schema_pass` 35/35,
    `complete` 35/35, zero argument/name hallucinations. Beats
    qwen2.5-7b (rubric 28/35, ref 28/35, fabs 8) on every metric
    at roughly one-third the footprint. Full row in
    [`eval/results/summaries/shape-d-2026-05-27.json`](https://github.com/rbentaarit/kubelm/blob/main/eval/results/summaries/shape-d-2026-05-27.json).

## Out-of-scope

- Training general K8s-domain models that aren't specifically
  targeting K8sGPT MCP. The trajectories' tool list is
  K8sGPT-specific; using these to teach `kubectl` directly will
  cause mode confusion.
- Use as a benchmark in itself — for evaluation, use the
  [Phase 1 eval harness](https://github.com/rbentaarit/kubelm/tree/main/eval)
  with the
  [Phase 2 scenario library](https://github.com/rbentaarit/kubelm/tree/main/eval/scenarios/specs).
  The trajectories here ARE solutions; evaluating on them is
  circular.
- Safety/refusal training. K8sGPT's architecture handles
  destructive operations through Mutation CRs + operator
  policy gates; the model is trained for tool-use reliability,
  not destructive-action gating.

## Limitations (please read)

1. **n = 29 unique scenarios.** The 365-record count comes from
   variation + negatives. The underlying *failure-mode coverage*
   is the 29 in the linked scenario library. A real production
   training corpus needs more failure modes — especially anything
   K8sGPT's MCP surface doesn't expose well today (e.g.,
   NetworkPolicy support is missing in v0.4.32, so policy-related
   investigations are constrained).

2. **All positives generated by gpt-5.4.** No model diversity. A
   future v0.2 should include trajectories from other strong
   models (Claude, multiple OpenAI variants, qwen2.5:32b) to
   broaden investigation-style coverage.

3. **Synthetic negatives have templated recovery prose.** A
   model trained on these may pick up the exact phrasing of the
   recovery assistant turns ("That resourceType isn't supported.
   The right value..."). Reviewers flagged these as
   `unreviewed`; treat the negatives as a starting point rather
   than ready-to-train data.

4. **Tools list is currently null in `tools` field.** The eval
   harness doesn't yet persist the K8sGPT MCP `tools/list`
   payload alongside trajectories; a helper script
   (`data/seed/snapshot_tools.py`) captures them per K8sGPT
   version but hasn't been run for this release. Trainers that
   need the tool schema can re-run the snapshot themselves or
   reconstruct it from K8sGPT v0.4.32's documentation.

5. **Grounding analyzer (v1) is brittle to structured prose.**
   The 2026-05-12 audit found that the v1 grounding metric
   flagged most of gpt-5.4's conclusions as ungrounded because
   the model paraphrases tool output in YAML-path / quoted /
   dotted-status format the substring matcher can't reconcile.
   The audit confirmed these are formatting artifacts, not
   genuine fabrications, and stamped `grounding_failed_v1_artifact:
   true` on the affected seeds. See
   [PROJECT.md decisions log 2026-05-12](https://github.com/rbentaarit/kubelm/blob/main/PROJECT.md)
   for the audit and the grounding-metric-v2 followup.

6. **Single seed runs.** No multi-seed variance estimate. If
   you re-run the gpt-5.4 generation against the same scenario
   library you may get different conclusions for some scenarios
   that the model treats as having multiple valid investigation
   paths.

## Citation

```
@misc{kubelm_seed_v0,
  title  = {kubelm seed trajectories v0},
  author = {Ramzi Ben Taarit and contributors},
  year   = {2026},
  url    = {https://huggingface.co/datasets/rbentaarit/kubelm-seed-v0},
  note   = {Generated against K8sGPT v0.4.32}
}
```

## Source code

All generation, conversion, variation, negative-synthesis, and
review code lives in the project repo:
[github.com/rbentaarit/kubelm](https://github.com/rbentaarit/kubelm),
under `data/seed/`. The 30-scenario library that produced the seeds
is in `eval/scenarios/specs/`. The eval harness is in `eval/`.

## v0.1 corpus (2026-05-20)

A second seed cut for the v0.1 training iteration. Motivation: the
Stage 5 benchmark (`eval/results/summaries/shape-c-2026-05-20.json`)
showed kubelm-edge-v0's only remaining gap to qwen2.5:7b is
`ref_pass` (reference-call discipline) — rubric, grounding (v2), and
narrative-consistency are already at parity. v0.1 attacks that gap
through **data**, not recipe.

Two changes from the v0 corpus:

1. **Two generator styles.** v0 used gpt-5.4 only (limitation #2
   above). v0.1 adds **qwen2.5-7b — the reference target itself**
   (`ref_pass 32/33` on the Stage 5 cut). Training the 1.5B student
   directly on the reference's tool-selection behavior is the
   targeted attack on the gap.
2. **Calibrated v2-grounding selection.** Every trajectory is
   filtered on the Stage 3 grounding-v2 metric
   (`grounding_v2_has_fabrication: false`) in addition to
   rubric-pass / schema-pass / termination-complete. No trajectory
   where the generator itself fabricated enters training. This
   supersedes review.py's v1 grounding heuristic (which left clean
   trajectories `unreviewed`).

| File | Records | Source | Pass filter |
|---|---|---|---|
| `v01/gpt-5.4-2026-05-20.jsonl` | 31 | Stage 5 bench (gpt-5.4) | 30 |
| `varied/v01/gpt-5.4-2026-05-20-varied.jsonl` | 310 | 10× variants | 300 |
| `v01/qwen2.5-7b-2026-05-20.jsonl` | 25 | Stage 5 bench (qwen2.5-7b) | 20 |
| `varied/v01/qwen2.5-7b-2026-05-20-varied.jsonl` | 250 | 10× variants | 200 |

**Training set after filter: 550 records across 32 of 33 scenarios.**
(`pod-insufficient-cpu-001` has no clean seed — the same scenario
excluded from v0; neither generator produced a rubric-passing,
fabrication-free trajectory for it.) Both generators cover 19
scenarios with distinct investigation styles.

Selection criteria (config `training/configs/kubelm-edge-v01.yaml`):
`conclusion_rubric_passed` + `schema_passed` +
`termination_label == complete` + `grounding_v2_has_fabrication ==
false`. No `review_status` gate — these are eval-harness trajectories
machine-validated against the calibrated v2 metric rather than the
v0-era human/heuristic review. That is a deliberate methodology
choice for the iteration, documented here.

Generated against K8sGPT `0.4.32`, MCP protocol `2025-03-26`, the
same pins as v0. The trajectories carry the same schema_version 1
record format.

## v0.2 corpus (2026-05-22)

The corpus `kubelm-edge-v0.3` was trained on. Two changes from v0.1:

1. **System prompt swap.** Every record's `system_prompt` is
   replaced with the corrected canonical prompt (kubelm-edge's
   inference-time `DEFAULT_SYSTEM_PROMPT` from
   [`eval/runner/loop.py`](https://github.com/rbentaarit/kubelm/blob/main/eval/runner/loop.py)).
   The v0/v0.1 corpora carried an older, under-specified prompt that
   stopped at top-level status (e.g. "Pending") rather than drilling
   to the root cause. The corrected prompt instructs resource-aware
   drill-down (workload→Pods, PVC→StorageClass, scheduling→nodes
   and taints) and explicit anti-fabrication wording. Applied via
   [`data/seed/bake_system_prompt.py`](https://github.com/rbentaarit/kubelm/blob/main/data/seed/bake_system_prompt.py),
   which preserves provenance and re-runs the same record format
   conversion as v0.1.
2. **Corrective seed for `pod-insufficient-cpu-001`.** The v0 and
   v0.1 corpora excluded this scenario because no generator
   (gpt-5.4, qwen2.5-7b) produced a clean rubric-passing,
   fabrication-free trajectory for it — the kind cluster's control-
   plane node was still NotReady at scenario setup time, so the
   scheduler parked the Pod on a transient `node.kubernetes.io/not-
   ready` taint instead of surfacing the intended `Insufficient cpu`
   verdict. The harness was hardened (`kind create --wait 90s` plus
   a `message_contains` settle matcher targeting the
   `PodScheduled=False` condition), then qwen2.5:32b was used to
   generate a single rubric-passing corrective trajectory + 10
   variants for this scenario. **v0.2 is the first corpus that
   covers all 33 scenarios from the contemporary library** —
   v0/v0.1's "32 of 33" gap is closed.

| File | Records | Source | Pass filter |
|---|---|---|---|
| `v02/gpt-5.4-2026-05-20.jsonl` | 31 | v01 corpus, prompt swapped | 30 |
| `varied/v02/gpt-5.4-2026-05-20-varied.jsonl` | 310 | v01 corpus, prompt swapped | 300 |
| `v02/qwen2.5-7b-2026-05-20.jsonl` | 25 | v01 corpus, prompt swapped | 20 |
| `varied/v02/qwen2.5-7b-2026-05-20-varied.jsonl` | 250 | v01 corpus, prompt swapped | 200 |
| `v02/pod-insufficient-cpu-corrective-2026-05-22.jsonl` | 1 | corrective bench (qwen2.5:32b) | 1 |
| `varied/v02/pod-insufficient-cpu-corrective-2026-05-22-varied.jsonl` | 10 | 10× variants | 10 |

**Training set after filter: 561 records across all 33 scenarios**
(selection criteria same as v0.1: `conclusion_rubric_passed` +
`schema_passed` + `termination_label == complete` +
`grounding_v2_has_fabrication == false`). Same schema_version 1
record format. Same K8sGPT v0.4.32 / MCP 2025-03-26 pins.

The training configuration used by `kubelm-edge-v0.3` is
[`training/configs/kubelm-edge-v02-qwen35.yaml`](https://github.com/rbentaarit/kubelm/blob/main/training/configs/kubelm-edge-v02-qwen35.yaml)
— Qwen3.5-2B base, identical corpus to
`training/configs/kubelm-edge-v02.yaml`. Both configs trained off
the same 561 records; the Qwen3.5-2B run shipped (see "Models
trained on this dataset" above), the Qwen2.5-1.5B run overfit and
was not released.

## Changelog

- **v0.2 (2026-05-22):** Third seed cut. System prompt swapped to
  the corrected `DEFAULT_SYSTEM_PROMPT` across every record;
  `pod-insufficient-cpu-001` covered via a new qwen2.5:32b
  corrective seed + 10 variants (the harness was hardened with
  `kind create --wait 90s` so this scenario is now solvable). 561
  records after filter, all 33 contemporary scenarios covered.
  K8sGPT v0.4.32.
- **v0.1 (2026-05-20):** Second seed cut for the v0.1 training
  iteration. +gpt-5.4 (31) +qwen2.5-7b (25) seeds + 560 variants
  from the Stage 5 bench against the 33-scenario library; 550 pass
  the v2-grounding filter. Adds a second generator style and
  calibrated-v2 fabrication filtering. K8sGPT v0.4.32.
- **v0 (2026-05-13):** Initial release. 29 seeds + 290 variants +
  46 negatives = 365 trajectories. K8sGPT v0.4.32.
