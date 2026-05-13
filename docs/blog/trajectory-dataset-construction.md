# Building kubelm's first trajectory training corpus

The kubelm project's [scenario-methodology blog](./scenario-methodology.md)
described the evaluation harness and the benchmark behind kubelm's
"why fine-tune at all?" answer. This post is the next layer up: how
the first training dataset was constructed, what got committed, and
what's deliberately deferred to v0.2.

The dataset itself sits in the repo at `data/seed/` and is published
as `rbentaarit/kubelm-seed-v0` on Hugging Face. K8sGPT pin: `0.4.32`.
MCP protocol: `2025-03-26`. 365 trajectories total. CC BY 4.0.

This post explains how those numbers came to exist.

---

## The shape of the question

The eval harness produces a stream of artifacts per benchmark run:
one `results.json` summarizing reliability metrics, one
`trajectory.jsonl` recording every assistant turn and every tool
result, per (model, scenario) pair. Those JSONLs are *almost*
training data — they have the system prompt, the goal, the tool
calls, the tool results, the conclusion. But they're not directly
usable for SFT, for two reasons.

First, the eval JSONL is optimized for *run reproduction*: it has
timestamps, latencies, per-step metadata, internal eval-harness IDs.
A trainer doesn't want any of that; it dilutes the training signal
and inflates file sizes. The first design choice was to define a
distinct training-trajectory format that strips run metadata down
to provenance only.

Second, the eval JSONL has no quality signal embedded. A trajectory
that errored at step 3 looks identical in shape to one that
concluded correctly — the verdict lives in the separate `results.json`.
Bundling the quality block into the training record means downstream
filters (`if record["quality"]["conclusion_rubric_passed"]:`) don't
need to chase a second file.

The training format lives at `data/seed/FORMAT.md`. Schema version 1.
Key shape decisions:

- **JSONL, one trajectory per line.** Compatible with HF TRL,
  Axolotl, Unsloth, and every other SFT framework that accepts
  JSONL.
- **OpenAI tool-use `messages` array.** The dominant wire format for
  tool-use training; using anything else means an adapter on every
  consumer. `tool_calls.function.arguments` is a JSON-encoded string
  (the OpenAI wire shape), not a parsed dict.
- **`provenance` and `quality` as separate top-level objects.**
  Downstream code can `if "messages" in record:` without dragging
  metadata into the model input. Trainers that want to filter on
  quality flags or stratify by source read those fields explicitly.
- **Schema-versioned.** A future v2 can add a citation field, a
  reasoning-trace field, or move to a different vendor's wire format
  by bumping `schema_version: 2` without breaking v0 consumers.

---

## Reusing the eval bench as a strong-model generator

ROADMAP Phase 4 step 1 said "generate seed trajectories from the
Phase 2 scenarios using a strong model with careful prompting,
manual review of each trajectory for correctness." Reading that
fresh, the natural interpretation is "run a separate generation
pass against the scenarios."

The 2026-05-12 Shape B benchmark had already done this. gpt-5.4
ran the 30-scenario library end to end; 29 of 30 trajectories
passed the conclusion rubric. Those are exactly the kind of clean,
multi-step, reference investigations the trajectory dataset wants
as seeds. The eval bench had already paid the cost.

The construction step became a back-conversion: take the eval
results, project them into the training format, and stamp
provenance pointing at the source run. That's `data/seed/convert.py`:

```bash
uv run python data/seed/convert.py \
    --bench eval/results/benchmarks/<bench_id>/summary.json \
    --model gpt-5.4 \
    --out data/seed/v0/gpt-5.4-2026-05-12.jsonl
```

29 records out. Provenance source: `eval_bench`. Generator model:
`gpt-5.4`. License: `CC-BY-4.0`. Review status: `unreviewed` — to
be stamped in the next step.

This re-framing matters for a reason that isn't obvious at first:
**every published baseline run has now become a training-data
source.** The scenario library, the bench harness, and the seed
corpus aren't three independent assets to maintain; they're three
views of the same underlying artifact (a model's investigation
against a real K8sGPT MCP cluster). When the scenario library grows
to 50 or 100 scenarios in a future phase, the next bench cut
produces another seed batch the same way.

---

## Review: rule-based + a token-level grounding heuristic

The 29 seeds came in with `conclusion_rubric_passed: true` because
the conversion filtered to rubric-passing runs by default. But
"rubric passed" is one check; the review checklist in
`data/seed/REVIEW.md` has more: schema validity, plausible tool-call
order, no errored calls being ignored, concise conclusion, no
out-of-scope fix prescriptions, no genuine ungrounded facts.

The first five (schema, termination, rubric) are pulled directly
from the eval harness's metric blocks. They're hard rules; any
failure flips review_status to `rejected`.

Grounding is more interesting. The eval harness's v1 grounding
analyzer flagged gpt-5.4 at 30/30 grounding-failure on the
2026-05-12 cut — every conclusion had at least one claim it
couldn't substring-match against tool results. That looked like a
clean "frontier model hallucinates supporting detail" signal, until
we audited it. (See the
[2026-05-12 audit in PROJECT.md](../../PROJECT.md) for the full
retraction.)

What the audit found: gpt-5.4 wasn't fabricating; it was rendering
tool output in dotted-path / YAML / quoted formats the substring
matcher couldn't reconcile with the raw text. `configMapKeyRef.name:
app-settings` was flagged ungrounded because the tool result had
`{"configMapKeyRef": {"name": "app-settings"}}` — same fact,
different rendering.

The review script (`data/seed/review.py`) implements that audit
mechanically. For each fact flagged ungrounded by the v1 analyzer:

1. Normalize (lowercase, strip whitespace/quotes/brackets).
2. Try plain substring match against the trajectory's concatenated
   tool-result text.
3. Try after stripping a dotted-path prefix
   (`foo.bar.baz: value` → `value`).
4. Try after dropping inner quotes (`"http-port"` → `http-port`).
5. Fall back to a multi-token containment check: split the fact on
   structural punctuation, drop stopwords and short tokens, require
   every remaining token to be present in the tool text.

If at least 60% of the trajectory's flagged facts pass that match,
the v1 grounding failure is annotated as a formatting artifact
(`grounding_failed_v1_artifact: true`) and the trajectory remains
eligible for `accepted`. Otherwise it stays `unreviewed` for human
inspection.

A first pass without the token-level fallback flagged 12 of the 29
seeds as needing human review. Inspection showed almost all the
"unmatched" facts were values present in tool output but punctuated
differently (`=` vs `:`, JSON braces, etc.). After the token-level
fallback, 29 of 29 went to `accepted`. The residual unmatched facts
are clearly benign: derived states (`NotReady` from `Ready=False`),
reasonable string composition (`http://<pod>:80/healthz` from probe
spec + port), tool-name self-references (the model recommending a
tool the user could call), the known K8sGPT MCP networkpolicies gap
(`default-deny` paraphrased from a policy name K8sGPT doesn't
expose).

Two lessons from that pass that go beyond this dataset:

- **The v1 grounding analyzer needs a v2.** Cross-model grounding
  comparisons aren't currently reliable. A verbose-but-faithful
  model loses the column; a terse model wins; neither result
  reflects truth.
- **Rule-based metrics need adversarial audit against
  verbose-but-faithful models.** When the bench reports a striking
  signal, audit before publishing. We caught this before pushing
  the gpt-5.4-grounding narrative externally.

Both are tracked in ROADMAP as Phase 3 followups.

---

## Generalization variation: namespaces and resource names

29 seeds aren't enough to train against. Every record carries the
exact identifiers from the kind cluster — `scenario-configmap-missing-001`,
`api-pod`, `data-pvc`. Training on those, a model would learn the
strings rather than the structural pattern of each failure mode.

The variation pipeline (`data/seed/vary.py`) produces N variants
per seed by substituting two dimensions:

- **The namespace.** `scenario-<scenario_id>` → from a 30-name
  pool of realistic K8s namespace strings (`prod-api`,
  `payments-svc`, `data-pipeline-prod`, `team-alpha`, etc.).
- **The primary resource names.** `api-pod`, `data-pvc`, `api-svc`,
  the various deployment/statefulset/cronjob names — replaced from
  role-specific pools.

Substitutions are consistent within a trajectory: every occurrence
across the system prompt, the goal, every assistant turn, every
tool call's argument JSON, every tool result's content (including
JSON-stringified bodies, including the nested
`last-applied-configuration` annotations that Kubernetes embeds in
metadata), and the final conclusion uses the variant strings. A
spot-check confirmed nested annotations rename correctly — the
substitution is end-to-end.

A few details that matter:

- **Substitution order is length-descending.** A longer key can't
  be a suffix of a shorter substitution that runs first. Otherwise
  `scenario-pod-crashloop-001` would get corrupted by an earlier
  `pod-crashloop` substitution.
- **For source keys ≤ 4 chars, word boundaries are honored.**
  Otherwise `api` (the label value) would corrupt `apiVersion`,
  `apps/v1`, and other unrelated tokens. The longer keys substitute
  as plain substrings since they're already specific enough.
- **The variant value must not contain the source as a prefix.**
  An early run mapped `db` → `db-init`; the variant `db-init` then
  contained `db` as a word-boundary token, and a downstream
  leakage check flagged it as a missed substitution. The picker
  now skips variants that share a prefix with short source keys.
- **Variants are deterministic given (scenario_id, variant_index).**
  Same input file + same N → byte-identical output, supporting
  reviewable diffs.

5 variants per seed proved out the pipeline (145 trajectories);
scaling to 10× brought the corpus to 290 variants. A word-boundary
leakage scan over the full 290 found 0 original-string occurrences
in the training payload.

---

## Negative examples: error → recovery

A model trained only on clean trajectories never sees what happens
when a tool call goes wrong. Without exposure to MCP error responses
and recovery patterns, a small model's behavior on the first bad
call in production is whatever its base-model priors happen to be.

The negative-example pipeline
(`data/seed/synthesize_negatives.py`) injects a *bad first call +
K8sGPT-shape error + a brief recovery turn* at the front of an
otherwise clean trajectory. The rest of the trajectory continues as
in the seed. Two patterns in v0:

- **`wrong_resource_type`** (17 trajectories). The bad call is
  `list-resources(resourceType="<plausible-typo>", ...)` —
  `pod_name` instead of `pods`, `service_list` instead of
  `services`, etc. The error string returned is reproduced verbatim
  from what K8sGPT actually emits for unsupported types (the audit
  caught the exact phrasing). Recovery: a short assistant message
  acknowledging the typo and the correct type, then the original
  trajectory.
- **`hallucinated_tool_name`** (29 trajectories). The bad call is
  `get-pod(namespace=...)` — a plausible-looking non-existent tool
  (the real tool is `get-resource`). This is a common small-model
  error mode: fusing the resource type into the tool name. MCP
  responds with `unknown tool: get-pod`. Recovery names the real
  tool and continues.

Both error strings use the K8sGPT MCP response shape verbatim:

```json
{
  "content": [{"type": "text", "text": "unsupported resource type: pod_name. Supported types: [...]"}],
  "isError": true
}
```

The recovery prose is the weakness. It's templated:

> "That resourceType isn't supported. The right value for this
> lookup is `pods` — retrying with the correct type."

A model trained on 46 examples of that exact phrasing will pick it
up verbatim. For v0 the negatives carry `review_status: "unreviewed"`
to flag this; v0.2 will either rotate phrasings deterministically
or rewrite them by hand.

---

## What v0 ships

Concrete inventory:

| file | records | source | review |
|---|---|---|---|
| `v0/gpt-5.4-2026-05-12.jsonl` | 29 | `eval_bench` (gpt-5.4) | 29 accepted |
| `varied/v0/gpt-5.4-2026-05-12-varied.jsonl` | 290 | `eval_bench_variation` (10× per seed) | inherited accepted |
| `varied/v0/negatives-2026-05-12.jsonl` | 46 | `negative_synthetic` (2 patterns) | 46 unreviewed |
| **Total** | **365** | | |

~9.3 MB on disk. K8sGPT v0.4.32. CC BY 4.0. Mirror on Hugging Face
at `rbentaarit/kubelm-seed-v0`.

Field-by-field: see `data/seed/FORMAT.md`. Review process: see
`data/seed/REVIEW.md`. Per-file orientation: see
`data/seed/README.md`. HF-style description: see
`data/seed/DATASET_CARD.md`.

---

## Limitations (please read before training on this)

These are real. v0 is a starting point, not a finished training
corpus.

1. **n = 29 unique scenarios.** The 365-record count comes from
   variation + negatives, which expand surface form, not failure
   mode coverage. A model trained on this will see 29 underlying
   investigations, each in many namespaces with many resource names.
   For broad coverage, the next iteration needs more scenarios
   (Phase 2 followup) and more failure modes the current K8sGPT MCP
   surface doesn't reach (NetworkPolicy support upstream).

2. **All positives generated by gpt-5.4.** No model-style diversity.
   gpt-5.4's particular style is terse (mean step count: 2.2) and
   structured (YAML-path / dotted-status formatting). A v0.2
   should include trajectories from Claude, gpt-4o, qwen2.5:32b to
   broaden the investigation-style coverage and avoid teaching the
   target small model to mimic one generator's tics.

3. **Synthetic negatives have templated recovery prose.** All 46
   recovery turns follow a near-identical phrasing. Reviewers
   flagged them `unreviewed` for exactly this reason. Treat them
   as a starting signal, not ready-to-train data.

4. **Tools list is currently `null` in records.** The eval harness
   doesn't yet persist K8sGPT MCP's `tools/list` payload alongside
   each trajectory. A `data/seed/snapshot_tools.py` helper captures
   them per K8sGPT version, but hasn't been run for this release.
   Trainers needing the tool schema can re-run the snapshot
   themselves; a future eval-harness change will persist tools
   inline in the trajectory's meta event.

5. **v1 grounding analyzer is brittle to structured prose.** See
   the audit in PROJECT.md decisions log 2026-05-12. The
   `grounding_failed_v1_artifact` flag is set after auditing each
   trajectory, but the underlying metric isn't trustworthy for
   cross-model comparison. A v2 metric (LLM-judge or structural
   fact matcher) is on the followup list.

6. **Single seed runs.** No multi-seed variance estimate. The same
   scenario could plausibly produce different conclusions from
   gpt-5.4 across seeds; v0 captures one path each.

---

## What's next

In rough priority for v0.2 (after `kubelm-standard` is trained on
v0):

- **Hand-vary the negative-example recovery prose.** A small batch
  of distinct phrasings, rotated deterministically across the 46
  records. Cheap and removes a real overfitting risk.
- **Generator diversity.** Run Claude, gpt-4o, qwen2.5:32b against
  the scenario library. Each successful trajectory becomes another
  seed batch, then another variation set. Convert + review + vary
  is already mechanical.
- **Hand-written trajectories for failure modes the synthesizer
  can't produce.** Multi-step exploration patterns where the first
  tool result is ambiguous and a second clarifying call is needed.
  The 29 seeds average 2.2 steps; the corpus is light on the
  3-to-6-step investigations that the harder real-world scenarios
  involve.
- **Negative patterns beyond v0's two.** Premature conclusion +
  reopen, wrong-namespace + correct, get-logs on a Pending pod +
  pivot to events, malformed JSON arguments + retry. Each pattern
  is another `synthesize_negatives.py` function.
- **Persist tools/list at eval time** so the converter doesn't need
  an out-of-band cache. Small `TrajectoryRecorder` change; the
  followup is logged in ROADMAP.
- **Grounding-metric v2.** LLM-judge or structural fact-matcher.
  Until this lands, grounding numbers aren't reliable enough to
  drive architectural claims about kubelm's thesis.

---

## How to use it

```python
from datasets import load_dataset
ds = load_dataset("rbentaarit/kubelm-seed-v0")

# Filter to high-quality positives for an initial SFT pass.
positives = ds["seeds"].filter(
    lambda r: r["provenance"]["review_status"] == "accepted"
    and r["quality"]["conclusion_rubric_passed"]
)
# 29 records — the rubric-passing gpt-5.4 trajectories.

# Or, include the variants for a larger training set with
# structural variety.
positives_with_variants = ds["seeds"].concatenate(ds["variants"])

# Or stratify by failure category by reading scenario_source_path.
# The Phase 2 library groups scenarios by failure mode in their ids.
```

The training payload (`messages`, `system_prompt`, `goal`, `tools`)
is what gets fed to the model. The `provenance` and `quality`
blocks are for filtering and analysis; they should never make it
into the SFT input. A simple drop-the-metadata transform before
batching is sufficient:

```python
def to_sft_input(record):
    return {"messages": record["messages"]}
```

The rest of the SFT pipeline — tokenization, chat-template
application, training-loop config — is framework-specific and out
of scope for this corpus.

---

## A note on small datasets

v0 is 365 trajectories. Most "fine-tuning datasets" published on
Hugging Face are an order of magnitude larger. We're starting
small for two reasons.

First, **every trajectory was reviewed**. The 29 seeds went through
the REVIEW.md checklist. The 290 variants are mechanical from those
seeds, so they inherit the review verdict. The 46 negatives are
explicitly flagged `unreviewed` to be honest about what isn't yet
trusted. A 10,000-trajectory dataset where 90% is auto-generated
and unreviewed is *worse* training signal than 365 trajectories
where every record's quality is known. Quality over volume isn't a
slogan here; it's an operational discipline.

Second, **the goal is `kubelm-standard` (3B) at production-grade
reliability on K8sGPT MCP tool use, not chat-model breadth**.
The behavior we want to teach is narrow: correct tool selection,
well-formed arguments, faithful grounding, sensible termination on
a specific MCP surface. Tasks that narrow can be taught from
hundreds of high-quality trajectories, especially when the base
model already has the underlying capability and just needs to be
oriented toward this surface.

That's the bet, anyway. Phase 5 will tell us whether it lands.
