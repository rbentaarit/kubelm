# Trajectory training-data format (v0)

This file defines the on-disk format for kubelm's training trajectories.
The format is deliberately distinct from the eval trajectory JSONL
(see `eval/trajectory.py`), which optimizes for run reproduction and
metric calculation. The training format optimizes for:

- Direct ingestion into supervised fine-tuning loops (one JSONL line
  per training example, no nested context to reconstruct).
- Provenance the way Hugging Face datasets want it (model that
  generated the trajectory, K8sGPT version, scenario it derives from,
  review status, license, source URL).
- Forward-compatibility: schema_version-pinned so a later v1 can
  add fields without breaking v0 consumers.

A training corpus is a directory of JSONL files. Each line is a
complete training trajectory. We commit only small curated seeds to
git under `data/seed/`; the larger generated corpora live on
Hugging Face per the ROADMAP Phase 4 deliverable.

---

## Top-level record

```json
{
  "schema_version": 1,
  "trajectory_id": "uuid4",
  "k8sgpt_version": "0.4.32",
  "mcp_protocol_version": "2025-03-26",
  "scenario_id": "configmap-missing-001",
  "scenario_source_path": "eval/scenarios/specs/configmap-missing-001.yaml",
  "provenance": { ... },
  "system_prompt": "...",
  "goal": "Why won't api-pod in scenario-configmap-missing-001 start?",
  "tools": [ { "name": "...", "description": "...", "input_schema": {...} } ],
  "messages": [ ... ],
  "quality": { ... }
}
```

### Field-by-field

| field | type | required | notes |
|---|---|---|---|
| `schema_version` | int | yes | Always `1` for this version. |
| `trajectory_id` | string (uuid4) | yes | Stable across re-conversions. |
| `k8sgpt_version` | string | yes | Pins the tool surface. Must match the `eval/trajectory.py` `K8SGPT_VERSION` at the time the source eval was run. |
| `mcp_protocol_version` | string | yes | Same. |
| `scenario_id` | string | yes | The Phase 2 scenario id this trajectory investigates. |
| `scenario_source_path` | string | yes | Repo path to the scenario YAML at trajectory time. Helps reviewers trace back. |
| `provenance` | object | yes | See below. |
| `system_prompt` | string | yes | Exact system prompt used to generate the trajectory. Required because tool-use behavior depends on it. |
| `goal` | string | yes | The user-level question, from the scenario YAML. |
| `tools` | array | yes | Tool list the model saw at generation time. Each entry: `{name, description, input_schema}`. Captured at trajectory time, not looked up from K8sGPT MCP at training time. |
| `messages` | array | yes | See below. The actual training payload. |
| `quality` | object | yes | See below. |

### `provenance`

```json
{
  "source": "eval_bench",
  "source_run_id": "8dbb6f2d-49af-...",
  "source_bench_id": "5ffee982-103a-...",
  "generator_model": "gpt-5.4",
  "generator_backend": "https://api.openai.com/v1",
  "generated_at": "2026-05-12T02:36:28.262+00:00",
  "license": "CC-BY-4.0",
  "review_status": "unreviewed"
}
```

| field | values | notes |
|---|---|---|
| `source` | `eval_bench`, `handwritten`, `negative_synthetic` | How this trajectory came to be. |
| `source_run_id` | uuid or null | If `source == "eval_bench"`, the eval run this was extracted from. |
| `source_bench_id` | uuid or null | If part of a bench summary, the bench id (cross-link to `eval/results/benchmarks/<id>/summary.json`). |
| `generator_model` | string | The model that actually produced the trajectory. For handwritten examples, set to `"human"`. |
| `generator_backend` | string | API base URL or `"local-author"`. |
| `generated_at` | ISO timestamp | When the trajectory was originally produced (not when this file was written). |
| `license` | string | Per ROADMAP commitment 6: dataset is CC-BY-4.0. |
| `review_status` | `unreviewed`, `accepted`, `accepted_with_edits`, `rejected` | See `REVIEW.md`. |

### `messages`

A list of role-tagged turns. Compatible with OpenAI tool-use message
shape so existing SFT loops can ingest without remapping:

```json
[
  { "role": "system", "content": "<system_prompt verbatim>" },
  { "role": "user", "content": "<goal verbatim>" },
  { "role": "assistant", "content": "", "tool_calls": [{"id":"call_1","type":"function","function":{"name":"analyze","arguments":"{\"namespace\":\"...\"}"}}] },
  { "role": "tool", "tool_call_id": "call_1", "content": "<JSON-encoded tool result>" },
  { "role": "assistant", "content": "", "tool_calls": [...] },
  { "role": "tool", ... },
  { "role": "assistant", "content": "<final conclusion text>" }
]
```

Notes:

- The `system` and `user` messages are derivable from the top-level
  `system_prompt` and `goal` but ARE duplicated into messages so
  the array is directly trainable without reconstruction.
- `tool_calls.function.arguments` is a JSON-encoded *string*, matching
  OpenAI's wire format. Consumers that need a parsed dict can
  `json.loads()` it.
- `tool` messages carry the raw tool result content as a string. If
  the MCP server returned structured content (the usual case), it is
  JSON-stringified.
- The final assistant message has `content` (the conclusion) and no
  `tool_calls`.

### `quality`

```json
{
  "termination_label": "complete",
  "schema_passed": true,
  "schema_name_halluc": 0,
  "schema_arg_halluc": 0,
  "reference_calls_passed": true,
  "conclusion_rubric_passed": true,
  "grounding_failed": false,
  "grounding_failed_v1_artifact": true,
  "step_count": 5,
  "model_latency_ms": 18954.1
}
```

| field | source | notes |
|---|---|---|
| `termination_label` | eval `termination_report.label` | `complete`, `premature`, `errored`, `truncated`. |
| `schema_passed` | eval `schema_report` | All tool calls schema-valid. |
| `schema_name_halluc` | eval `schema_report.name_hallucinations` | Should be 0 in a good seed. |
| `schema_arg_halluc` | eval `schema_report.argument_hallucinations` | Should be 0 in a good seed. |
| `reference_calls_passed` | eval `reference_calls_report.passed` | True if the trajectory hit the scenario's any_of. |
| `conclusion_rubric_passed` | eval `conclusion_rubric_report.passed` | The single most important quality signal. |
| `grounding_failed` | eval `grounding_report.has_grounding_failure` | v1 metric; may be noisy. |
| `grounding_failed_v1_artifact` | reviewer-annotated | If `true`, the v1 grounding failure was a formatting/structural artifact (see PROJECT.md decisions log 2026-05-12), not a genuine hallucination. Reviewer can set this during `REVIEW.md` walkthrough. |
| `step_count` | derived | Number of assistant turns. |
| `model_latency_ms` | eval `totals.model_latency_ms` | Sum across all assistant calls in this trajectory. |

A trajectory is **eligible as a positive seed** when:
- `termination_label == "complete"`
- `schema_passed == true`
- `conclusion_rubric_passed == true`
- `provenance.review_status` is `accepted` or `accepted_with_edits`

Negative seeds (Phase 4 plan item #3) will use the same schema but
with `provenance.source == "negative_synthetic"` and a free-form
`note` describing what was wrong.

---

## Why this format

- **JSONL with one trajectory per line.** Most SFT toolchains
  (Hugging Face TRL, Axolotl, Unsloth) accept JSONL natively.
- **OpenAI-shaped `messages` array.** The dominant tool-use training
  payload shape; minimizes adapter code.
- **Provenance and quality as separate top-level objects.** Keeps the
  training payload (`messages`, `tools`, `system_prompt`) clean
  enough to feed directly into a loop with `if "messages" in record:`.
- **Schema-versioned.** Future Mac changes (e.g., adding a citation
  field, swapping `tool_calls` for a different vendor's wire format)
  can land as `schema_version: 2` without breaking v0 consumers.

## What's NOT in the format (and why)

- **Model logits, top-k tokens, anything sampling-side.** Not needed
  for SFT; would balloon file size 100x.
- **Reasoning traces / chain-of-thought.** Not produced by the eval
  harness, and we don't want to commit to a CoT-flavored training
  recipe in v0.
- **Per-step latency breakdown.** Aggregate in `quality`; per-step
  belongs in the eval trajectory, not the training one.
- **K8sGPT raw analyzer output beyond what was passed back to the
  model.** The model only saw the MCP tool results; the training
  shouldn't have privileged information.
