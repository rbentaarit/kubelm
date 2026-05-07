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
