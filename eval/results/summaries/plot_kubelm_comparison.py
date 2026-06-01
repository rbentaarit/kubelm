"""Plot kubelm vs the cloud frontier vs generic local models on the
current native-v2 metrics (35-scenario suite).

Data sources (all native-v2 metrics):
  - gpt-5.4 (cloud frontier)        : shape-c-2026-05-20.json   (n=33)
  - qwen2.5-1.5b / -7b, edge-v0/v0.3: shape-d-2026-05-27.json   (n=35)
  - kubelm-edge-0.8b (1-epoch)      : kubelm-0.8b-finetune-2026-05-29.json

gpt-5.4 was run over 33 scenarios, the rest over 35, so the rubric panel
uses pass RATES (normalized); fabrications is a raw count (noted).

    uv run --group viz python eval/results/summaries/plot_kubelm_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent


def agg(path: str, want: set[str]) -> dict[str, dict]:
    d = json.loads((HERE / path).read_text())
    out: dict[str, dict] = {}
    for r in d["runs"]:
        m = r["model"]
        if m not in want:
            continue
        a = out.setdefault(m, dict(rubric=0, fabs=0, n=0))
        a["n"] += 1
        a["rubric"] += 1 if r.get("conclusion_rubric_passed") else 0
        a["fabs"] += r.get("fabrications") or 0
    return out


gpt = agg("shape-c-2026-05-20.json", {"gpt-5.4"})["gpt-5.4"]
sd = agg(
    "shape-d-2026-05-27.json",
    {"qwen2.5-1.5b", "qwen2.5-7b", "kubelm-edge-v0", "kubelm-edge-v0.3"},
)

# (label, category, rubric_pass, n, fabs, footprint)
#   category: cloud | generic | kubelm | kubelm-hi
ROWS = [
    (
        "qwen2.5-1.5b\n(base, 0.9 GB)",
        "generic",
        sd["qwen2.5-1.5b"]["rubric"],
        35,
        sd["qwen2.5-1.5b"]["fabs"],
    ),
    ("kubelm-0.8b\n(0.5 GB)", "kubelm", 24, 35, 14),
    (
        "kubelm-edge-v0\n(1.5B, 0.9 GB)",
        "kubelm",
        sd["kubelm-edge-v0"]["rubric"],
        35,
        sd["kubelm-edge-v0"]["fabs"],
    ),
    (
        "kubelm-edge-v0.3\n(2B, 1.2 GB)",
        "kubelm-hi",
        sd["kubelm-edge-v0.3"]["rubric"],
        35,
        sd["kubelm-edge-v0.3"]["fabs"],
    ),
    ("qwen2.5-7b\n(4.7 GB)", "generic", sd["qwen2.5-7b"]["rubric"], 35, sd["qwen2.5-7b"]["fabs"]),
    ("gpt-5.4\n(cloud)", "cloud", gpt["rubric"], gpt["n"], gpt["fabs"]),
]

COLOR = {
    "generic": "#9aa7b4",
    "kubelm": "#e8833a",
    "kubelm-hi": "#c5521a",
    "cloud": "#4a6fa5",
}

labels = [r[0] for r in ROWS]
colors = [COLOR[r[1]] for r in ROWS]
rubric_rate = [100 * r[2] / r[3] for r in ROWS]
fabs = [r[4] for r in ROWS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.5))

b1 = ax1.bar(labels, rubric_rate, color=colors)
ax1.set_title(
    "Reasoning: conclusion-rubric pass rate\n(reaches the correct root cause, grounded)",
    fontsize=11,
)
ax1.set_ylabel("% of scenarios passed")
ax1.set_ylim(0, 105)
ax1.axhline(100 * gpt["rubric"] / gpt["n"], ls="--", lw=1, color="#4a6fa5", alpha=0.6)
for bar, v in zip(b1, rubric_rate, strict=True):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)

b2 = ax2.bar(labels, fabs, color=colors)
ax2.set_title(
    "Grounding: fabrications (lower is better)\nfacts asserted that no tool returned", fontsize=11
)
ax2.set_ylabel("fabrication count over the suite")
for bar, v in zip(b2, fabs, strict=True):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        v + max(fabs) * 0.01 + 0.5,
        str(v),
        ha="center",
        fontsize=9,
    )

for ax in (ax1, ax2):
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=c)
    for c in (COLOR["cloud"], COLOR["generic"], COLOR["kubelm"], COLOR["kubelm-hi"])
]
fig.legend(
    handles,
    ["cloud frontier", "generic local", "kubelm", "kubelm (headline)"],
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=9,
)

fig.suptitle(
    "kubelm vs the cloud frontier on K8sGPT MCP tool-use  (35-scenario suite, native v2 metrics)",
    fontsize=13,
    fontweight="bold",
)
fig.text(
    0.5,
    0.005,
    "gpt-5.4 over 33 scenarios (rate-normalized); others over 35. "
    "CPU-only, no GPU in the kubelm runtime path. "
    "All kubelm tiers: zero tool-name/argument hallucinations.",
    ha="center",
    fontsize=8,
    color="#555",
)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))

out = HERE / "kubelm-vs-frontier-2026-05-31.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
