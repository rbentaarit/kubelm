"""Render the hallucination-vs-size chart from a Shape B summary JSON.

Usage:
    uv run --group viz python eval/results/summaries/plot_shape_b.py \
        eval/results/summaries/shape-b-2026-05-07.json

The output PNG lands next to the input JSON, with the same basename.

This is a deliberately minimal artifact: one figure, three series
(reference-call pass, conclusion-rubric pass, grounding-failure rate)
across the model size axis. Cloud frontier models are placed on the
right with a separator, since their effective parameter count is not
public.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

LOCAL_SIZES_B = {
    "llama3.2-3b": 3.0,
    "qwen2.5-7b": 7.0,
    "qwen2.5-32b": 32.0,
}
CLOUD_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.5",
}


def main(summary_path: Path) -> Path:
    data = json.loads(summary_path.read_text())
    summaries = data["model_summaries"]
    n_scenarios = len(data["scenarios"])

    locals_ordered = sorted(
        (m for m in summaries if m in LOCAL_SIZES_B),
        key=lambda m: LOCAL_SIZES_B[m],
    )
    clouds_ordered = [m for m in summaries if m in CLOUD_MODELS]

    def pct(model: str, key: str) -> float:
        return 100.0 * summaries[model][key] / n_scenarios

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    x_local = list(range(len(locals_ordered)))
    x_cloud = list(range(len(locals_ordered) + 1, len(locals_ordered) + 1 + len(clouds_ordered)))
    xs = x_local + x_cloud
    labels = [
        *(f"{m}\n({LOCAL_SIZES_B[m]:.0f}B)" for m in locals_ordered),
        *(f"{m}\n(cloud)" for m in clouds_ordered),
    ]
    models_ordered = locals_ordered + clouds_ordered

    ref_pass = [pct(m, "reference_calls_passed") for m in models_ordered]
    rubric_pass = [pct(m, "conclusion_rubric_passed") for m in models_ordered]
    ground_fail = [pct(m, "grounding_failures") for m in models_ordered]

    ax.plot(xs, ref_pass, marker="o", label="reference-call pass %")
    ax.plot(xs, rubric_pass, marker="s", label="conclusion-rubric pass %")
    ax.plot(xs, ground_fail, marker="x", linestyle="--", label="grounding-failure %")

    if x_local and x_cloud:
        sep = (x_local[-1] + x_cloud[0]) / 2
        ax.axvline(sep, color="gray", linewidth=0.5, linestyle=":")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of scenarios")
    ax.set_title(
        f"Shape B: reliability vs model size  "
        f"(n={n_scenarios} scenarios, K8sGPT {data['k8sgpt_version']})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out_path = summary_path.with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    out = main(Path(sys.argv[1]))
    print(out)
