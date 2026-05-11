# Phase 3 baseline benchmark — first results

These are the headline `summary.json` artifacts from the first benchmark
runs against the kubelm scenario library. Per-run trajectories live
elsewhere (gitignored under `eval/results/<run-id>/`); this directory
only carries the cross-model summaries that document the published
numbers.

Each file is a self-contained record: scenarios used, model lineup
(`backend_url`, `model`, `temperature`, `max_tokens`), `k8sgpt_version`,
`mcp_protocol_version`, `cluster_strategy`, `parallelism`, every
per-`(model × scenario)` run with its five-metric report, and per-model
totals.

## 2026-05-07 — Shape A: `llama3.2:3b` vs `gpt-4o-mini`

Smoke run that proved the pipeline. 2 models × 10 scenarios = 20 runs,
~15 min wall-time on M1 Max 64 GB.

| model | complete | schema | ground_fail | ref_pass | rubric_pass | errored |
|---|---|---|---|---|---|---|
| llama3.2-3b | 0/10 | 10/10 | 9 | 0/10 | 1/10 | 0 |
| gpt-4o-mini | 10/10 | 10/10 | 5 | 3/10 | 8/10 | 0 |

**File:** `shape-a-2026-05-07.json`

## 2026-05-07 — Shape B: 4-model size curve

3B → 7B → 32B → cloud frontier. 4 models × 10 scenarios = 40 runs,
~32 min wall-time on M1 Max 64 GB. (Llama 3.3 70B was the original
"large local" target but OOM'd alongside kind/Docker — qwen2.5:32b is
the local stand-in until a GPU-box benchmark fills in the 70B point.)

| model | complete | schema | ground_fail | ref_pass | rubric_pass | duration_s |
|---|---|---|---|---|---|---|
| llama3.2-3b | 0/10 | 10/10 | 9 | 0/10 | 1/10 | 295 |
| qwen2.5-7b | 10/10 | 10/10 | 6 | 5/10 | 5/10 | 454 |
| qwen2.5-32b | 10/10 | 10/10 | 5 | 4/10 | 5/10 | 878 |
| gpt-4o | 10/10 | 10/10 | 6 | 6/10 | 6/10 | 303 |

**File:** `shape-b-2026-05-07.json`

![Shape B: reliability vs model size](shape-b-2026-05-07.png)

Regenerate with:

```
uv run --group viz python eval/results/summaries/plot_shape_b.py \
    eval/results/summaries/shape-b-2026-05-07.json
```

### Headline findings

1. **3B → 7B is a phase change.** The 3B model cannot drive a
   multi-step investigation against this surface (`complete` 0/10).
   At 7B and above, every model reaches `complete` 10/10. There is a
   capability cliff somewhere in that interval.
2. **Above 7B the curve is essentially flat.** qwen2.5-7b, qwen2.5-32b,
   and gpt-4o all land in the 5–6/10 rubric range with similar
   grounding-failure counts. Adding parameters past 7B produces no
   measurable improvement on this surface.
3. **Schema is clean across all 30 successful 7B+ runs.** Zero
   tool-name and zero argument hallucinations. The failure modes are
   strategic (when to call what, when to stop), not syntactic.
4. **gpt-4o's edge over qwen2.5:7b is small.** 6/10 vs 5/10 on ref
   and rubric. The frontier reference is competent but not
   meaningfully better than a free 4.7 GB Qwen model on this task.

### Caveats

- **n = 10 scenarios** — small sample. Single-scenario differences
  are noise; the 3B-vs-rest gap is the only signal that's robust.
- **Rubric noise** — the 4–6/10 cluster on `ref_pass`/`rubric_pass`
  partly reflects scenario-rubric strictness, not just model
  competence. Iteration on matchers is ongoing.
- **No 70B local point.** The ROADMAP "rented GPU box" remains the
  proper home for the 70B and 32B+ tier; the local 32B run is a
  stand-in that establishes the curve shape, not a replacement for
  the GPU benchmark.
- **`parallelism: 1`** — these are valid latency numbers per the
  parallel-vs-serial protocol in `docs/blog/scenario-methodology.md`.

## 2026-05-11 — Shape B: 5-model size curve (rubric v2, gpt-5.4)

Refresh of the 2026-05-07 baseline. Two changes vs the prior run:

1. **Rubric v2** — synonym-slot iteration from commit `3d31af5` is
   applied uniformly to all models. The same trajectory now scores
   ~1 rubric point higher across the lineup, so the v2 numbers are
   not directly comparable to the 2026-05-07 cells.
2. **Cloud frontier upgraded.** `gpt-5.4` joins as the latest OpenAI
   model that still supports `temperature: 0` (gpt-5 and gpt-5.5
   reject deterministic sampling — see commit message of the
   `_uses_max_completion_tokens` backend change). `gpt-4o` stays in
   the lineup for continuity with the prior baseline.

5 models × 10 scenarios = 50 runs. Two scenarios errored on transient
infra issues (qwen2.5-32b read timeout, gpt-4o rate-limit 429) and are
counted as `errored: 1` in their rows, not as a model competence loss.

| model | complete | schema | ground_fail | ref_pass | rubric_pass | errored | duration_s |
|---|---|---|---|---|---|---|---|
| llama3.2-3b | 0/10 | 10/10 | 9 | 0/10 | 2/10 | 0 | 314 |
| qwen2.5-7b | 10/10 | 10/10 | 5 | 4/10 | 6/10 | 0 | 504 |
| qwen2.5-32b | 9/10 | 9/10 | 3 | 4/10 | 6/10 | 1 | 984 |
| gpt-4o | 9/10 | 9/10 | 7 | 6/10 | 7/10 | 1 | 334 |
| gpt-5.4 | 10/10 | 10/10 | 10 | 6/10 | 9/10 | 0 | 360 |

**File:** `shape-b-2026-05-11.json`

![Shape B: reliability vs model size (2026-05-11)](shape-b-2026-05-11.png)

Regenerate with:

```
uv run --group viz python eval/results/summaries/plot_shape_b.py \
    eval/results/summaries/shape-b-2026-05-11.json
```

### Headline findings

1. **The 3B cliff holds.** llama3.2-3b is still `complete: 0/10`,
   rubric 2/10 (was 1/10 — the +1 is the rubric-v2 shift, not model
   improvement). The capability gap below 7B is the most robust
   signal across both runs.
2. **gpt-5.4 beats the local frontier on the rubric.** 9/10 vs the
   6–7/10 cluster from 7B–32B local and gpt-4o. This is the first
   run where the cloud frontier is meaningfully ahead of mid-size
   local models on conclusion quality.
3. **gpt-5.4 has the worst grounding score.** 10/10 scenarios show
   at least one ungrounded claim in the conclusion — the highest in
   the lineup, despite the highest rubric pass-rate. The frontier
   model reaches the *right answer* while reasoning from
   context-not-tool-output. n = 10 is too small to conclude;
   investigate per-scenario before drawing. One drill-in already
   done — see the `network-policy-block-001` note below.
4. **Schema is clean across all 48 successful runs.** Zero tool-name
   and zero argument hallucinations across the full lineup. Same
   finding as 2026-05-07 — failure modes are strategic, not
   syntactic.
5. **7B → 32B is still flat.** qwen2.5-7b and qwen2.5-32b score
   identically on rubric and ref_pass. The local size curve plateaus
   below 70B. (qwen2.5-32b dropped one scenario to a 120s read
   timeout on `service-selector-mismatch-001`, which slightly
   understates it.)

### Drill-in: `network-policy-block-001` (gpt-5.4, rubric fail)

This is the single scenario gpt-5.4 failed (9/10 rubric). The drill-in
turned up two issues with the bench, not the model:

- **K8sGPT MCP tool gap.** `list-resources` only supports
  `[ingress, persistentvolumeclaim, persistentvolume, pod, deployment,
  service, cronjob, daemonset, configmap, secret, node, job,
  statefulset, replicaset]`. NetworkPolicy is not on that list. The
  model correctly called `list-resources(resourceType=networkpolicies)`
  and got `unsupported resource type` back. The policy name
  `default-deny-ingress` was never available in any tool result, but
  the scenario's rubric requires the model to name it. That's an
  unsatisfiable requirement on the current K8sGPT MCP surface.
  *Action:* file upstream against K8sGPT; either relax this rubric
  or skip the scenario in the meantime.
- **`ref_pass=True` despite tool error.** The reference-calls metric
  counted the failing `list-resources(networkpolicies)` call as
  "reference call passed" because the argument shape matched the
  scenario's `any_of`, regardless of whether the MCP server actually
  returned data. That's a bench bug — `ref_pass` should require a
  non-errored tool result. *Action:* gate the matcher on
  `isError != true` before publishing the Phase 3 blog.

The model itself behaved reasonably: it tried the right tool, fell
back to inferring "NetworkPolicy blocks ingress" from the scenario
name + goal text when the tool refused, and got the diagnosis right
without naming the policy. This explains a meaningful share of
gpt-5.4's 10/10 grounding-failure count: the "ungrounded" claims are
often the model filling gaps that K8sGPT couldn't fill, not the model
inventing facts it could have looked up. The "frontier model
hallucinates supporting detail" story needs to wait until the bench
bug is fixed and the rubric is reconciled with the tool surface.

### Caveats

- **Two transient errors.** qwen2.5-32b's 120s timeout on
  `service-selector-mismatch-001` and gpt-4o's 429 on
  `resource-quota-block-001` are infra issues, not methodology
  signal. Re-running the two affected (model × scenario) pairs would
  push their `errored` counts to 0 without changing the others.
- **Rubric v2 ≠ rubric v1.** The 2026-05-07 row and the 2026-05-11
  row use different rubric matchers. Compare cells *within* one
  table, not *across* the two tables.
- **gpt-5.4 grounding finding is preliminary.** 10 scenarios is
  noisy. The grounding analyzer is rule-based; a frontier model may
  trip rules that smaller models don't simply because it writes more
  prose. The `network-policy-block-001` drill-in (above) already
  shows that at least some of gpt-5.4's grounding failures are
  filling gaps the tool surface couldn't fill, not inventing
  lookupable facts. Audit per-scenario trajectories before claiming
  this is a real model-level grounding gap.
- **`ref_pass` over-counts.** As described in the drill-in above,
  the metric currently passes if the tool-call argument shape
  matches the scenario's `any_of`, even when the tool returned an
  error. This affects the published `ref_pass` cells for at least
  network-policy-block-001 and possibly other scenarios where the
  reference call hits a K8sGPT MCP unsupported-type response. Fix
  before publishing the Phase 3 blog.
- **No 70B point** (same caveat as 2026-05-07).
- **Wall-time** of this run was inflated by host sleep — bench
  process is correct but the start→end timestamps span hours of
  idle, not active compute. The per-run `duration_seconds` column
  is the authoritative latency record.
