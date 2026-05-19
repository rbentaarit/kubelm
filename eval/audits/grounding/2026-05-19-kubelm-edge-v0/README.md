# Per-scenario grounding audit — kubelm-edge-v0 (2026-05-19)

## Why this audit exists

The 2026-05-14 v0 release shipped with a `grounding_failed` count of
**27/30**, vs the base qwen2.5:1.5b baseline of 16/30 and attempt-1's
21/30. The HF model card and PROJECT.md decisions log flag this
honestly as a documented metric caveat — the v1 grounding analyzer is
rule-based, doesn't tolerate structural rephrasing (YAML paths,
dotted-state notation, quoted/unquoted strings), and fine-tuning is
precisely the operation that shifts a model's output style. The
2026-05-12 audit of gpt-5.4's 30/30 grounding score retracted the
"frontier hallucinates supporting detail" narrative for exactly this
reason.

This audit applies the same per-scenario walkthrough to kubelm-edge-v0
before any external grounding claim is made.

## What gets classified

`prepare.py` walks every scenario's `results.json` + `trajectory.jsonl`
under v0's run directory and emits one record per ungrounded fact in
`audit.yaml`. Each record carries:

- `scenario_id`
- `fact` — the exact string flagged by the v1 grounding analyzer
- `conclusion_excerpt` — ~200 chars of the assistant's final
  conclusion around the fact (for context)
- `tool_results_searchable` — concatenation of every tool result the
  model received in this trajectory, so the auditor can grep for
  rephrased versions of the fact
- `classification` (empty initially) — one of the five labels below
- `rationale` (empty initially) — short note explaining the call

## Classification taxonomy

Borrowed verbatim from the 2026-05-12 gpt-5.4 audit:

| Label | Meaning | Example |
|---|---|---|
| `fabrication` | The fact is genuinely not derivable from any tool result. Real hallucination. The number worth reporting externally. | Model claims "the pod is OOMKilled" when no tool result mentions OOM. |
| `structural_rephrase` | The fact IS in tool results, but in a different syntactic form the v1 analyzer can't match. | Model writes `state.waiting.reason: CrashLoopBackOff`; tool result has `{"state":{"waiting":{"reason":"CrashLoopBackOff"}}}`. Same content, different render. |
| `composed_inference` | The fact is a reasonable string composition from primitives in tool results, not literally present in any single result. | Model writes `http://<pod>:80/healthz` composed from a probe spec `httpGet.path=/healthz` + container `port=80`. |
| `scenario_fill` | The fact comes from the goal statement or scenario context, not from tool calls. The trajectory recorder doesn't put the goal in the "tool results" pool, so it shows as ungrounded even though it was given to the model. | Model echoes the namespace name from the goal. |
| `unsupported_tool` | The fact is about something K8sGPT MCP v0.4.32 can't expose (NetworkPolicy names, ResourceQuota labels, etc.). Same gap that hit gpt-5.4 on `network-policy-block-001`. | Model names a NetworkPolicy `default-deny-ingress` from scenario context because `list-resources(networkpolicies)` errors. |

Only `fabrication` counts as a model defect. The other four are
either metric-blind-spots or expected behavior.

## Procedure

```bash
# 1. Build the working file from v0's run output (slim — no tool corpus).
#    The committed audit.yaml is this slim form (~40 KB) so it can live
#    in git. Each record has scenario_id, fact, conclusion_excerpt, and
#    empty classification + rationale fields.
uv run python eval/audits/grounding/2026-05-19-kubelm-edge-v0/prepare.py \
    --run-dir eval/results/checkpoints/kubelm-edge-v0-attempt-2 \
    --out eval/audits/grounding/2026-05-19-kubelm-edge-v0/audit.yaml

# 1b. (Optional) For offline grep, regenerate with the full tool-result
#     corpus inlined into each record (~3 MB; do NOT commit this form).
uv run python eval/audits/grounding/2026-05-19-kubelm-edge-v0/prepare.py \
    --run-dir eval/results/checkpoints/kubelm-edge-v0-attempt-2 \
    --out /tmp/audit-with-corpus.yaml \
    --include-corpus

# 2. Manually classify each entry. Open audit.yaml, fill in the
#    `classification:` and `rationale:` fields for every record.
#    To confirm a structural rephrasing rather than a fabrication,
#    grep against the per-scenario trajectory directly:
#       jq -r '.content.content[]?.text // empty' \
#           eval/results/checkpoints/.../<scenario>/trajectory.jsonl \
#           | grep -i 'pattern'
#    or use the --include-corpus form from step 1b.

# 3. Aggregate
uv run python eval/audits/grounding/2026-05-19-kubelm-edge-v0/summarize.py \
    eval/audits/grounding/2026-05-19-kubelm-edge-v0/audit.yaml
```

The summarizer prints a five-row table (fabrication / rephrase /
composed / scenario_fill / unsupported_tool) plus a per-scenario
breakdown, and writes a markdown summary that gets committed
alongside the audit YAML.

## What the audit changes

If the classification shows `fabrication` is a small minority of the
27 facts (the gpt-5.4 audit showed "roughly a handful" out of 30):

- PROJECT.md gets a new decisions log entry retracting the strict
  reading of v0's grounding number
- The HF model cards get an updated grounding section linking to
  this audit
- The `eval/results/summaries/README.md` 2026-05-14 section gets a
  pointer to the audit's conclusion

If `fabrication` is a meaningful share (>30% of the 27 facts):

- The grounding number stands as a real regression
- v0.1 needs a data iteration to address the specific failure modes
- The HF model cards get a stronger warning

Either outcome is publishable. The point of the audit is to know
which one is true.

## Reproducibility

This audit is reproducible from `eval/results/checkpoints/kubelm-edge-v0-attempt-2/`
which is gitignored but contains the exact per-trajectory records.
The audit YAML committed in this directory is the human-reviewed
output — the prepare.py script can regenerate the unclassified
working file at any time from the same input.

Same script can be run against attempt-1, against future training
iterations, or against other models in the same eval-output shape.
