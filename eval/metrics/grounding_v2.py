"""Grounding analyzer v2 — 5-label classifier.

Replaces the v1 boolean grounding analyzer (``eval/metrics/grounding.py``)
with one that emits one of the 5 labels from the Stage 2 audit taxonomy:

- ``fabrication`` — model invented the fact; not derivable from tool
  output, scenario context, or known K8s vocabulary.
- ``structural_rephrase`` — fact IS in tool corpus, just rephrased
  (JSON to dotted notation, quoted/unquoted, whitespace differences).
- ``composed_inference`` — fact tokens scattered through tool corpus;
  reasonable composition of primitives but not a literal substring.
- ``scenario_fill`` — fact text comes from the scenario goal or
  namespace name, not tool output.
- ``unsupported_tool`` — fact concerns a resource type K8sGPT MCP v0.4.32
  cannot expose (NetworkPolicy, ResourceQuota labels, etc.). Reserved
  category; rarely emitted (the v0 audit had zero of these).

Only ``fabrication`` counts toward the bench's headline
``fabrications_total``. The other four are recorded but not penalized —
they're metric blind-spots from the v1 era.

The classifier exposes both a per-fact ``classify`` method (used by the
calibration harness in ``eval/audits/grounding/calibrate.py``) and a
trajectory-level ``analyze`` function (used by the bench, mirroring the
v1 API). The trajectory-level entry point re-uses v1's fact extraction
so v1 and v2 are scored against the same fact universe per run — the
only thing that changes is the label.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from eval.metrics.grounding import analyze_grounding as v1_analyze_grounding

# K8sGPT MCP v0.4.32 cannot expose these resource types via
# `list-resources`. A fact that names one of these as a *resource type*
# (not as part of a namespace name) gets the `unsupported_tool` label.
UNSUPPORTED_RESOURCE_TYPES: frozenset[str] = frozenset(
    {
        "networkpolicies",
        "networkpolicy",
        "resourcequotas",
        "resourcequota",
        "limitranges",
        "limitrange",
        "podsecuritypolicies",
        "podsecuritypolicy",
    }
)

# Coverage thresholds tuned against the v0 audit (n=114, calibrate.py):
# - >= STRICT_THRESH: tokens fully or near-fully cover the corpus →
#   structural_rephrase. Set high so we don't claim rephrasing on
#   weakly-supported facts.
# - >= LOOSE_THRESH: enough tokens land that the model could have
#   composed the fact from primitives → composed_inference. The
#   space below is fabrication.
STRICT_COVERAGE_THRESH = 0.85
LOOSE_COVERAGE_THRESH = 0.5

# Tokens shorter than this are ignored for coverage analysis (too
# generic). Anything < 3 chars is mostly stop-words and noise.
SIGNIFICANT_TOKEN_LEN = 3

# Minimum length for an alphanumeric-squash match to count as a
# rephrase signal. Tuned against the v0 audit: 8 catches CamelCase↔
# hyphenated cases like ``NotReady`` matching ``not-ready`` in the
# corpus, but short fact strings like ``2/2`` (squash → ``22``) and
# ``NoGo`` (squash → ``nogo``) are below it so they fall through to
# the token-coverage rules.
MIN_SQUASH_LEN = 8


# --- normalization helpers ---


def _strip_json_punctuation(s: str) -> str:
    """Collapse JSON quoting/braces so structural rephrasing matches.

    `"key": "value"` becomes `key value`; backticks, square brackets,
    quotes all become spaces. Multiple whitespace collapsed to one
    space. Lower-cased.
    """
    s = re.sub(r'[\\"`\[\]{},:;]+', " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _alphanumeric_squash(s: str) -> str:
    """Even more aggressive: keep only [a-z0-9], drop everything else.

    Used as a fallback for facts that vary in punctuation (dotted-state
    paths, camelCase, etc.). Matches generously.
    """
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _significant_tokens(s: str) -> list[str]:
    """Tokens >= SIGNIFICANT_TOKEN_LEN, lowercased."""
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", s) if len(t) >= SIGNIFICANT_TOKEN_LEN]


def _scenario_context(scenario_id: str, goal: str) -> str:
    """Everything the model can read from the scenario's goal + namespace name."""
    return " ".join([scenario_id, goal, f"scenario-{scenario_id}"]).lower()


# --- classifier ---


@dataclass
class V2Classifier:
    """Per-fact 5-label classifier.

    Decision tree:
    1. Empty fact -> structural_rephrase (degenerate).
    2. Normalized literal match in corpus -> structural_rephrase.
    3. Alphanumeric-squash match in corpus -> structural_rephrase.
    4. Significant tokens entirely in corpus -> composed_inference if
       not contiguous, structural_rephrase if contiguous-ish.
    5. Resource-type token names an unsupported MCP type -> unsupported_tool.
    6. Tokens in scenario goal / namespace but not corpus -> scenario_fill.
    7. Token coverage somewhat high (>= COVERAGE_THRESH) -> composed_inference.
    8. Fallback -> fabrication.
    """

    name: str = "v2"

    def classify(
        self,
        *,
        fact: str,
        corpus: str,
        scenario_id: str,
        goal: str,
    ) -> str:
        fact = (fact or "").strip()
        if not fact:
            return "structural_rephrase"

        nf = _strip_json_punctuation(fact)
        nc = _strip_json_punctuation(corpus)

        # Rule 2: normalized literal substring match.
        if nf and nf in nc:
            return "structural_rephrase"

        # Rule 3: alphanumeric-squash match (handles JSON-nested ↔ dotted
        # rephrasing where punctuation differs heavily). Require a
        # minimum match length so short facts like "2/2" don't squash
        # to "22" and match anywhere.
        nf_a = _alphanumeric_squash(fact)
        nc_a = _alphanumeric_squash(corpus)
        if nf_a and len(nf_a) >= MIN_SQUASH_LEN and nf_a in nc_a:
            return "structural_rephrase"

        toks = _significant_tokens(fact)

        # Rule 5: unsupported resource type. Only fire when the token
        # appears AS a standalone resource word, not as part of a
        # composite namespace string like "resource-quota-block-001".
        for unsup in UNSUPPORTED_RESOURCE_TYPES:
            if unsup in toks:
                return "unsupported_tool"

        if not toks:
            # Fact has no significant tokens (e.g. "2/2", "8", "v1").
            # Version-tag-like patterns (`v1.2.4`, `1.27`) are a common
            # fabrication shape — the model recommends a hypothetical
            # tag as a fix. If the literal version string isn't in
            # corpus, flag as fabrication. Otherwise the empty-tokens
            # fallback is composed_inference (safer middle bucket for
            # short cluster-state shorthand like "2/2").
            if re.fullmatch(r"v?\d+(\.\d+){1,3}!?", fact.strip()) and fact.strip() not in corpus:
                return "fabrication"
            return "composed_inference"

        # Rule 4: token coverage. Note that we already checked literal
        # and alphanumeric-squash substring match above; reaching here
        # means even cov=1.0 facts had their tokens scattered through
        # corpus (e.g. condition-status pairs like "Available: False"
        # where each token is in the corpus but not next to each other).
        # Both cov=1.0 (no literal/squash) and cov >= STRICT cases
        # become composed_inference — the tokens are derivable but the
        # specific composition isn't a literal rephrase.
        miss = [t for t in toks if t not in nc]
        cov = (len(toks) - len(miss)) / len(toks)
        if cov >= STRICT_COVERAGE_THRESH:
            return "composed_inference"

        # Rule 6: scenario fill (goal / namespace text, not corpus). We
        # check this BEFORE composed_inference's looser threshold so
        # that a fact whose tokens are mostly goal-derived doesn't get
        # absorbed into composed.
        scen = _strip_json_punctuation(_scenario_context(scenario_id, goal))
        if nf and nf in scen:
            return "scenario_fill"
        scen_toks = set(_significant_tokens(_scenario_context(scenario_id, goal)))
        scen_hits = sum(1 for t in toks if t in scen_toks) if toks else 0
        if (
            toks
            and scen_hits / len(toks) >= STRICT_COVERAGE_THRESH
            and cov < STRICT_COVERAGE_THRESH
        ):
            return "scenario_fill"

        # Mid-range coverage: the model assembled the fact from
        # primitives in tool output, but not enough is literally
        # present to claim rephrasing. composed_inference, not
        # fabrication — fabrication requires real absence.
        if cov >= LOOSE_COVERAGE_THRESH:
            return "composed_inference"

        # Fallback: fabrication.
        return "fabrication"


# --- trajectory-level entry point ---


@dataclass
class FactClassification:
    fact: str
    label: str  # one of LABELS in calibrate.py


@dataclass
class GroundingV2Report:
    """Per-trajectory v2 report.

    Mirrors v1's ``GroundingReport`` but per-fact emits a label, and
    the headline boolean ``has_fabrication`` replaces v1's
    ``has_grounding_failure`` (which conflated all four "metric
    blind-spot" labels into the same alarm).
    """

    conclusion_text: str
    total_facts: int
    fabrications: int
    has_fabrication: bool
    facts: list[FactClassification] = field(default_factory=list)


def analyze_grounding_v2(events: Iterable[Mapping[str, Any]]) -> GroundingV2Report:
    """Score a trajectory's conclusion under the 5-label taxonomy.

    Reuses v1's fact extraction (so the v1 vs v2 comparison is over
    the same fact universe) and reuses v1's tool-corpus assembly via
    its internal helpers. Per-fact label comes from V2Classifier.
    """
    events_list = list(events)
    v1_report = v1_analyze_grounding(events_list)

    # Reconstruct the searchable corpus the same way prepare.py does.
    corpus_parts: list[str] = []
    for e in events_list:
        if e.get("kind") != "tool_result":
            continue
        content = e.get("content")
        if isinstance(content, dict):
            for c in content.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text":
                    corpus_parts.append(c.get("text", ""))
        elif isinstance(content, str):
            corpus_parts.append(content)
    corpus = "\n".join(corpus_parts)

    # Read scenario_id from the trajectory meta — bench writes it
    # alongside goal. If the recorder didn't capture either, the
    # scenario_fill rule won't fire (no false positives).
    meta = next((e for e in events_list if e.get("kind") == "meta"), {})
    scenario_id = meta.get("scenario_id", "") or ""
    goal = meta.get("goal", "") or ""

    classifier = V2Classifier()
    classifications: list[FactClassification] = []
    fab_count = 0
    for f in v1_report.facts:
        if f.grounded:
            # v1 already marked this grounded; v2 inherits.
            classifications.append(FactClassification(fact=f.fact, label="structural_rephrase"))
            continue
        label = classifier.classify(
            fact=f.fact,
            corpus=corpus,
            scenario_id=scenario_id,
            goal=goal,
        )
        if label == "fabrication":
            fab_count += 1
        classifications.append(FactClassification(fact=f.fact, label=label))

    return GroundingV2Report(
        conclusion_text=v1_report.conclusion_text,
        total_facts=len(classifications),
        fabrications=fab_count,
        has_fabrication=fab_count > 0,
        facts=classifications,
    )
