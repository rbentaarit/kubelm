# Benchmarking K8sGPT MCP tool-use: a scenario methodology

*Draft. The kubelm scenario library and the protocol it enables.*

Tool-use reliability — the rate at which a model calls the right tools
with well-formed arguments and faithfully grounds its conclusions — is
not measurable on real production clusters. Production clusters change
every minute. They contain real workloads with real owners, real RBAC
policies, real secrets, and real consequences for being wrong. You
cannot reproduce a benchmark run. You cannot share the data. You
cannot diff results between models or between K8sGPT versions because
the clusters underneath are not the same clusters.

So you build scenarios.

A scenario is a small, deterministic, well-known failing K8s cluster
plus the investigation a competent SRE should run on it. The model
under evaluation drives K8sGPT's MCP tools through that investigation;
the eval harness records what it does and grades it against the
scenario's expectations. You can run the same scenario tomorrow and
get the same numbers. You can publish the scenarios on GitHub. You can
diff this month's model release against last month's against the
exact same surface.

This post walks through how kubelm structures that machinery — the
scenario format, the profile composition, the cluster lifecycle, the
parallel-vs-serial protocol, and the five reliability metrics each
trajectory is graded against. It assumes you've read
[PROJECT.md](../../PROJECT.md) and
[ROADMAP.md](../../ROADMAP.md) for context on what kubelm is and
isn't.

---

## What a scenario looks like

`eval/scenarios/specs/pod-crashloop-001.yaml` is the canonical small
example:

```yaml
id: pod-crashloop-001
profile: base
description: |
  A Pod whose container prints a startup error and exits 1 immediately,
  triggering kubelet's restart loop until kubelet declares CrashLoopBackOff.

goal: "Why is the crash-pod in the scenario-pod-crashloop-001 namespace failing?"

setup:
  - apply_inline: |
      apiVersion: v1
      kind: Namespace
      metadata: { name: scenario-pod-crashloop-001 }
      ---
      apiVersion: v1
      kind: Pod
      metadata:
        name: crash-pod
        namespace: scenario-pod-crashloop-001
      spec:
        containers:
          - name: app
            image: busybox:1.36
            command: [sh, -c, 'echo "startup failed: missing CONFIG_PATH"; exit 1']

settle:
  - wait_for_status:
      kind: Pod
      namespace: scenario-pod-crashloop-001
      name: crash-pod
      reason: CrashLoopBackOff
      timeout: 90s

expected:
  reference_calls:
    must_include:
      - { name: list-resources, args_match: { resourceType: pods } }
      - { name: get-logs,       args_match: { podName: crash-pod } }
  conclusion_rubric:
    must_mention: ["CrashLoopBackOff", "crash-pod"]
    semantic_intent: |
      Identifies the pod, names the failure mode, and references the
      startup-error log line as the cause.
```

A scenario has six load-bearing parts:

- **`id`** — globally unique handle the runner uses for cluster naming,
  output paths, and result aggregation.
- **`profile`** — which cluster fixture to run against (more on this
  below).
- **`goal`** — the SRE-style question the model is given as user input.
  Real questions, not synthetic prompts.
- **`setup`** — manifests applied to the cluster after profile install
  to produce the failure. Either inline YAML or a sibling
  `apply_file:` reference.
- **`settle`** — wait conditions that confirm the failure has actually
  manifested before the model run begins. Without this, you race
  against kubelet and sometimes evaluate against a cluster where the
  symptom hasn't surfaced yet.
- **`expected`** — what the runner grades the trajectory against.
  `reference_calls.must_include` lists tool calls a competent
  investigation makes (with subset-semantic argument matchers, not
  exact sequences); `must_not_mention` and `must_mention` are
  case-insensitive substring rubrics on the final conclusion;
  `semantic_intent` is preserved for human review and v0.2 LLM-judge
  evaluation.

The format is one YAML file per scenario, parsed by
`eval/scenarios/spec.py:load_scenario`. Phase 4 (training-data
construction) iterates the same files programmatically. Single
declarative format > Python modules.

---

## Profiles: composable cluster fixtures

Different failure classes need different clusters. A pod-startup
failure runs against a barebones kind cluster. A GitOps-related
failure needs Argo CD pre-installed. A multi-cluster scenario needs a
specific networking add-on. Re-installing those operators per scenario
would multiply benchmark wall-time minutes by the number of scenarios
that share a fixture.

A profile is a YAML file describing the cluster-level state every
scenario in that profile inherits:

```yaml
name: argocd
extends: base
install:
  - helm_repo:    { name: argo, url: https://argoproj.github.io/argo-helm }
  - helm_install: { name: argocd, chart: argo/argo-cd, namespace: argocd, version: 7.6.12 }
wait_for:
  - { kind: Deployment, namespace: argocd, name: argocd-server, condition: Available, timeout: 3m }
prebuilt_image: kubelm-argocd-base:v1
```

Profiles compose by single inheritance (`extends:`). At run time,
`compose_profile()` flattens the chain root-first so parent install
steps run before child install steps; scalar fields cascade
child-wins-when-set. Cycles and unknown extends targets raise.

`prebuilt_image:` is the optimization layer: rather than running
`install:` at every cluster create, you can bake the resulting state
into a kind node image once and reuse it across runs. v0.1 ships the
install path; image baking is documented for Phase 3 when total
benchmark time becomes load-bearing.

---

## The determinism floor: per-scenario fresh kind cluster

The single biggest design call in Phase 2: every scenario gets its
own freshly-created kind cluster.

Sharing a cluster across scenarios was tempting. Operators take
minutes to install and stabilize; reinstalling Argo CD twenty times to
run twenty argocd-profile scenarios is forty wasted minutes. With
namespace-per-scenario isolation, the savings looked clean.

They are not clean. K8s is a state-storing distributed system, and the
state that leaks across scenarios is not a corner case:

- **K8sGPT's MCP server is stateful.** The `add-filters` and
  `remove-filters` tools mutate the analyzer process. Scenario A's
  model calls `add-filters: ["Pod"]` and the next scenario starts with
  Pod-filtered analysis it didn't ask for.
- **K8sGPT's analyzer cache.** Cached analyses from scenario A surface
  in scenario B.
- **Cluster-scoped resources.** ClusterRoles, CRDs, StorageClasses,
  ValidatingWebhookConfigurations, PriorityClasses — none respect
  namespace boundaries.
- **Operator state survives namespace deletion.** Argo CD's Application
  controller, Flux's reconciler, kube-prometheus's ServiceMonitors are
  all cluster-wide and outlive their namespace.
- **Node-level state.** OOMKilled pods leave eviction records; later
  scenarios pick them up.
- **Order dependence.** Scenario 1 in fresh-cluster passes; scenario 50
  in same cluster after 49 prior scenarios sometimes fails because of
  accumulated drift. The benchmark becomes order-dependent and
  non-reproducible by anyone else who runs it.

That last bullet is the entire reason a benchmark exists. A run that
nobody else can reproduce isn't a benchmark; it's an anecdote.

This is also how serious agent benchmarks are built. SWE-Bench runs
each instance in a fresh Docker container at a specific commit.
AgentBench gives each task its own Compose stack. WebArena's full-env
reset between tasks is a designed-in operation, not an afterthought.
τ-Bench restarts state per conversation. The pattern is unanimous;
kubelm follows it.

The cost is real but bounded: ~25–35 minutes of cluster lifecycle for
50 scenarios serialized, dropping toward ~10 minutes with prebuilt
operator images. For a once-per-release published number, that's
fine. For interactive iteration while *authoring* scenarios — where
30 seconds per cluster is painful — the runner exposes a
`--cluster-strategy shared` dev-mode escape that reuses one cluster
across scenarios. Results from that mode are tagged
`"determinism": "shared-cluster-dev-mode"` and refused for
`eval/results/benchmarks/`. You cannot accidentally publish a
contaminated number.

---

## Parallel-vs-serial: a protocol, not a setting

If every scenario gets its own cluster, the obvious next move is to
parallelize. With deterministic per-scenario state, scenarios are
embarrassingly parallel. RAM, disk I/O, and Docker daemon throughput
are the only ceilings. A 64GB workstation runs 30+ kind clusters
concurrently; cloud CI runs whatever you pay for.

But: per-step latency is one of kubelm's primary metrics. CPU
performance under load is a *primary design constraint* — the whole
project is about reliable tool-use on commodity CPU hardware.
Parallel execution introduces shared CPU/RAM contention that distorts
latency measurements: a model that takes 12 seconds per step
serialized may take 25 seconds per step when ten siblings are
hammering the same machine. The reliability metrics are unaffected;
the latency numbers become uninformative.

So kubelm runs benchmarks in two passes:

**Pass 1 — parallel, reliability-only.** `parallelism=N` (where N is
whatever the host can sustain). Records hallucination rates, grounding
failures, termination labels, reference-call coverage, and
conclusion-rubric pass/fail. Fast.

**Pass 2 — serial, latency-only.** `parallelism=1`. Records per-step
latency, total trajectory wall-time, and total tokens. Slow but valid.

`results.json` carries `parallelism: int` so reviewers can tell at a
glance whether latency numbers in a published run are trustworthy.
The Phase 3 baseline benchmark — when it lands — will publish both
passes side-by-side.

The runner interface is designed for this from day one. All shared
state (cluster names, kubeconfigs, ports, output dirs, helm caches)
is derived from `(run_id, scenario_id)` so any parallelism factor
works without changes. v0.1 ships `parallelism=1`; the seam is real.

---

## The five metrics

Each trajectory produces a `results.json` with five graded reports.
The first three carry across all eval runs (Phase 1); the latter two
are scenario-specific (Phase 2).

1. **Schema validation.** Every recorded `tool_call.arguments` is
   checked against the tool's advertised JSON Schema. Splits cleanly
   into *tool-name hallucinations* (calls to nonexistent tools) and
   *argument hallucinations* (known tool, args fail schema).
2. **Grounding analysis.** The final assistant text is parsed for
   K8s-shaped factoids (kebab-case identifiers, image refs, status
   reasons) and each factoid is checked for verbatim presence in the
   user's goal or some prior tool result. Any unsupported assertion
   marks the trajectory as a grounding failure. v0.1 is rule-based;
   v0.2 will add an LLM-judge variant for paraphrase / negation /
   quantity claims.
3. **Termination classification.** The trajectory shape determines a
   single label: `errored`, `no_conclusion`, `looping`, `premature`,
   or `complete`. Looping is verbatim
   `(tool_name, arguments)` repetition (semantic loops are v0.2).
   Premature is conclusion-with-zero-successful-tool-calls.
4. **Reference-call coverage** *(Phase 2)*. The scenario's
   `must_include` matchers (subset-semantic on arguments) all hit at
   least once; `any_of` matchers (at least one of N) are satisfied; no
   `forbidden` matcher hits. The `any_of` semantics capture the
   "multiple valid investigation paths" reality — `get-logs` and
   `list-events` both surface a pod failure cause; either is fine.
5. **Conclusion rubric** *(Phase 2)*. The scenario's `must_mention`
   strings all appear (case-insensitive substring) in the final
   assistant text; no `must_not_mention` does. `semantic_intent` is
   preserved on the report for human review and v0.2 LLM-judge.

Each report is independent. A trajectory that calls every reference
tool but reaches a wrong conclusion fails the rubric and passes
reference-calls. A trajectory that loops on the right tools with
correct args fails termination and may still pass reference-calls.
This separation is deliberate — they fail in different ways and
matter to different users.

---

## What v0.1 ships, and what it doesn't

**Eight scenarios** at
`eval/scenarios/specs/`, covering the v0.1 coverage targets from
ROADMAP.md Phase 2: pod startup (`pod-crashloop-001`,
`image-pull-001`), resource (`oom-killed-001`,
`resource-quota-block-001`), service connectivity
(`service-selector-mismatch-001`, `network-policy-block-001`), RBAC
(`rbac-denied-001`), storage (`pvc-unbound-001`).

**One profile** (`base`); the `argocd` profile is shipped as a worked
example of `extends:` composition but is not used by any v0.1
scenario.

**No paid-API runs.** The Phase 3 benchmark — actually running models
against this library — is the next milestone. v0.1 of the scenario
library is the substrate it sits on.

**No semantic rubric evaluation.** The `semantic_intent` field is
preserved but not auto-graded. Substring matching catches the
egregious cases; v0.2's LLM-judge will close the long tail.

**No prebuilt operator images.** Documented as the v0.2 optimization;
ships when it actually saves minutes on a real benchmark run.

---

## First baseline results (2026-05-07)

The library above produced its first cross-model numbers. Two passes
on a single M1 Max 64 GB at `parallelism=1` (so latency is trustworthy
but the dataset is tiny — n=10 scenarios, single seed).

**Shape B — 4 models across the size spectrum** (`eval/results/summaries/shape-b-2026-05-07.json`):

| model         | complete | schema | ground_fail | ref_pass | rubric_pass | duration_s |
|---------------|----------|--------|-------------|----------|-------------|------------|
| `llama3.2:3b` | 0/10     | 10/10  | 9           | 0/10     | 1/10        | 295        |
| `qwen2.5:7b`  | 10/10    | 10/10  | 6           | 5/10     | 5/10        | 454        |
| `qwen2.5:32b` | 10/10    | 10/10  | 5           | 4/10     | 5/10        | 878        |
| `gpt-4o`      | 10/10    | 10/10  | 6           | 6/10     | 6/10        | 303        |

`llama3.3:70b` was the original local-large target but OOM'd on a
64 GB M1 Max with the bench's kind clusters running concurrently
(~42 GB model + KV cache + ~5 GB harness > free memory). `qwen2.5:32b`
is the local stand-in until a proper GPU-box benchmark fills in the
70B point.

The data tells a sharper story than the original PROJECT.md thesis
predicted:

**1. The 3B → 7B step is a phase change, not a curve.** The 3B model
cannot drive a multi-step investigation against this surface at all —
zero `complete` terminations across ten scenarios. At 7B and above,
every model reaches `complete` 10/10. Whatever capability is required
to "see a tool result, decide what to call next, recognize when to
stop and synthesize" emerges somewhere between 3B and 7B and either
exists or doesn't.

When you watch a 3B trajectory, the failure is concrete: the model
confuses the namespace name with a pod name on the first call, gets a
"not found" back, and concludes the pod doesn't exist. It then tries
to call `get-events` *by writing JSON inside the text body* rather
than emitting a structured tool_call. Tool-use infrastructure is
present; tool-use *competence* is not.

**2. Above 7B, the curve is essentially flat.** `qwen2.5:7b`,
`qwen2.5:32b`, and `gpt-4o` cluster at 5–6/10 on rubric, 4–6/10 on
reference-call coverage, 5–6 grounding failures out of 10. The 32B
model does not measurably outperform the 7B model. The cloud frontier
reference is barely better than a free 4.7 GB local model. **More
parameters past 7B produce no measurable improvement on this task.**

**3. Schema is clean across all 30 successful 7B+ runs.** Zero
tool-name hallucinations, zero argument hallucinations. None of the
failures are syntactic. All of them are strategic — *what* to call,
*when* to stop, *whether* to keep investigating after the first
result.

### What this implies for kubelm's thesis

The original PROJECT.md framing assumed the curve sloped smoothly: as
parameters increase, tool-use reliability improves, and a specialized
small model would close the gap to a large generic model. The data
suggests something different. There is no smooth curve to close. There
is a cliff between 3B and 7B, and a flat plateau above 7B.

This actually *strengthens* the case for kubelm:

- The capability gap to close is between a 3B and a 7B-class model.
  That is exactly the size range where local-CPU inference is most
  valuable (sub-3 GB vs. ~5 GB Q4 weights). A specialized 3B that
  matches 7B-generic on K8sGPT MCP tool-use would deliver outsized
  practical value.
- Spending compute to fine-tune at the 32B–70B range has low expected
  payoff, because generic models in that range are already competent
  enough that the rubric/grounding gap looks more like noise than
  signal.

It also raises a question we couldn't answer before: *is the 7B-class
plateau the actual ceiling for this task, or is it the rubric's
ceiling?* The 4–6/10 cluster on `ref_pass`/`rubric_pass` may partly
reflect scenario-rubric strictness. Iteration on matchers and a
larger scenario set will tell.

### Caveats

These numbers are a v0.1 baseline, not a publication-ready benchmark.

- **n = 10 scenarios.** Statistically too small to read a single-row
  difference into. The 3B-vs-rest gap is the only signal robust enough
  to lean on.
- **Single-seed runs.** No variance estimate. Re-running the same
  configuration could land 1–2 metrics apart on any given row.
- **Rubric noise.** The 4–6/10 cluster above 7B partly reflects
  matcher strictness. The ongoing rubric iteration may compress that
  range up or down.
- **No 70B local point.** Important data missing from the curve. The
  ROADMAP "rented GPU box" remains the right home for it.
- **No pre-Llama-3 family models.** The whole lineup is recent
  instruction-tuned models trained with tool-use in mind.

A real publication needs n ≥ 30 scenarios, multiple seeds per
configuration, and the 70B point. These results are the methodology
working — not the final number.

---

## Contributing scenarios

The most useful contribution today is *more scenarios*. The format is
deliberately simple to author. If you've debugged a real K8s issue at
work that would make a good benchmark — pick a representative failure
mode, write a single YAML file describing the failing manifests + the
expected investigation, drop a PR.

Things that make a good scenario:

- A failure mode that surfaces in a specific status field or event,
  not "things are slow today."
- An investigation a competent SRE could run in fewer than ten
  K8sGPT MCP tool calls.
- A conclusion that can be expressed as a small set of must-mention
  strings without leaning on prose semantics.
- Manifests that fit on a screen.

Things that don't:

- Failures that require minutes of stabilization to surface.
- Failures whose ground truth depends on cluster history that scenarios
  can't reproduce.
- Investigations that require tools K8sGPT's MCP surface doesn't
  expose. (If the tool surface is missing something genuinely
  necessary, contribute to K8sGPT — kubelm follows that surface, it
  doesn't shadow it.)

The Phase 3 benchmark will run every scenario in the library. New
scenarios go into the next published run.
