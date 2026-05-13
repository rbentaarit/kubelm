# Trajectory review checklist (v0)

Every seed trajectory has `provenance.review_status: "unreviewed"`
when it lands from the converter. Before it can be used as a positive
training example, a human reviewer walks the trajectory and stamps it
`accepted`, `accepted_with_edits`, or `rejected`. This file is the
checklist the reviewer follows.

A trajectory is **accepted as-is** when every box below is checked.
Any unchecked box is justification for `accepted_with_edits` (note
what was edited and why) or `rejected` (note the failure mode).

---

## How to review one trajectory

1. Open `data/seed/v0/<file>.jsonl`.
2. Find the record by `trajectory_id` (or `scenario_id` — they're
   1:1 for the converter's bench output mode).
3. Read top-to-bottom. The `messages` array is the substance; the
   `quality` block is the eval harness's read of it.
4. Run the checklist below.
5. Update `provenance.review_status`, and for grounding-flagged
   trajectories, set `quality.grounding_failed_v1_artifact` to
   `true` / `false` based on the audit (see #5).
6. If editing the conclusion, set `provenance.review_status` to
   `accepted_with_edits` and add a `provenance.review_notes` field
   describing what changed.

A simple editor pattern is `jq` to read + a script to update in
place; future tooling may add `data/seed/review.py` for guided
review.

---

## Checklist

### 1. Goal and system prompt

- [ ] `goal` matches the scenario's `goal:` field in
      `eval/scenarios/specs/<scenario_id>.yaml` verbatim.
- [ ] `system_prompt` is the standard kubelm SRE prompt
      (i.e., not silently customized for this trajectory).

### 2. Tool calls

- [ ] All tool names are in the K8sGPT MCP surface for the pinned
      `k8sgpt_version` (cross-check with `data/seed/tools/<ver>.json`
      once that file exists).
- [ ] No name hallucinations (`quality.schema_name_halluc == 0`).
- [ ] No argument hallucinations
      (`quality.schema_arg_halluc == 0`).
- [ ] The tool-call sequence is *plausible*: a competent SRE
      investigating this scenario would make similar calls. Cluster
      enumeration → resource detail → events → logs (if relevant)
      is the standard arc; deviations should have a reason.
- [ ] No unnecessary repeated calls (e.g. `list-resources(pods)`
      called twice with identical args).

### 3. Tool results

- [ ] No tool result was an MCP error
      (`isError: true` in the result content) that the model then
      ignored. An errored call followed by a sensible pivot is
      fine; an errored call followed by a conclusion that depends
      on the missing data is not.
- [ ] Where the scenario requires a specific tool result to be
      observed, that result is in the trajectory (cross-check
      against the scenario's `expected.reference_calls.must_include`
      / `any_of` matchers).

### 4. Conclusion

- [ ] `messages[-1]` is the assistant's final text-only turn (no
      `tool_calls`).
- [ ] `conclusion_rubric_passed` is `true`. If `false`, the
      trajectory needs an edited conclusion or should be rejected.
- [ ] Conclusion is concise (≤ ~10 sentences ideally). Verbose
      conclusions train verbose models.
- [ ] No fix prescription that goes beyond the scenario (e.g.,
      a database migration script when the scenario is about a
      missing ConfigMap reference). Stick to *identify the cause*,
      *suggest the minimal fix*.

### 5. Grounding (per the 2026-05-12 audit)

If `quality.grounding_failed: true`, the v1 grounding analyzer
flagged at least one claim in the conclusion as not present in any
tool result. The 2026-05-12 audit (see PROJECT.md decisions log)
showed that for verbose models like gpt-5.4, most flagged facts are
*actually* in tool output but rendered in a format the analyzer
can't match (YAML-path notation, quoted vs unquoted, dotted status
paths, reasonable string composition).

For each flagged trajectory:

- [ ] Open `eval/results/<source_run_id>/<scenario_id>/results.json`
      and read the `grounding_report.facts` list.
- [ ] For every fact where `grounded: false`, check whether the
      claim is in fact derivable from a tool result in this
      trajectory. If yes (formatting mismatch): set
      `quality.grounding_failed_v1_artifact = true`.
- [ ] If any flagged fact is genuinely not in tool output AND
      not a reasonable inference from scenario context (e.g., a
      composed URL), the trajectory should be `rejected` or have
      its conclusion edited.

### 6. Provenance

- [ ] `generator_model` and `generator_backend` look right
      (no `unknown` placeholders).
- [ ] `license` is `CC-BY-4.0` (Apache 2.0 for code, CC-BY-4.0 for
      data per PROJECT.md commitment #6).
- [ ] If you edited the trajectory, set `review_status` to
      `accepted_with_edits` and add a `review_notes` field
      describing what changed (e.g., "rewrote conclusion to drop
      out-of-scope fix prescription").

---

## Rejection criteria (any one is sufficient)

- Tool name or argument hallucination
  (`schema_name_halluc > 0` or `schema_arg_halluc > 0`).
- Conclusion doesn't satisfy the scenario's rubric and can't be
  cheaply edited to (e.g., the model went down a wrong tool path
  early and the conclusion follows from a wrong premise).
- Genuine ungrounded fact in the conclusion that's not a v1 metric
  artifact and not a reasonable inference from scenario context.
- Conclusion is so verbose or hedged that it would teach the model
  bad output style.

---

## Bulk review status

Use:

```
jq -c '.provenance.review_status' data/seed/v0/<file>.jsonl | sort | uniq -c
```

…to see how much of a file is still `unreviewed`. The target is
`unreviewed = 0` for any file referenced from the Hugging Face
dataset card.
