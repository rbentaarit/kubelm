# `data/seed/` — Phase 4 trajectory training data

Small, curated training corpora committed to the repo. The larger
production corpora live on Hugging Face per the Phase 4 deliverable;
this directory is where they're authored, reviewed, and packaged
before upload.

## File layout

```
data/seed/
├── FORMAT.md                 schema_version 1 spec (one trajectory per JSONL line)
├── REVIEW.md                 per-trajectory review checklist
├── DATASET_CARD.md           HF-style README to ship with the dataset
├── README.md                 this file
│
├── convert.py                eval results → training trajectory JSONL
├── vary.py                   generalization variation pipeline (namespace + resource renames)
├── synthesize_negatives.py   negative examples (error → recovery)
├── review.py                 auto-review pass; stamps review_status
├── snapshot_tools.py         capture K8sGPT MCP tools/list per version
│
├── tools/                    (created by snapshot_tools.py)
│   └── <k8sgpt_version>.json
│
└── varied/                   variant + negative corpora (mechanical)
    └── v0/
        ├── gpt-5.4-2026-05-12-varied.jsonl       290 trajectories
        └── negatives-2026-05-12.jsonl             46 trajectories
```

And:

```
data/seed/v0/
└── gpt-5.4-2026-05-12.jsonl                    29 reviewed seed trajectories
```

## Pipeline order

1. **Run the eval bench** against the scenario library with one or
   more strong models (see `eval/scenarios/benchmarks/`). The
   2026-05-12 cut already produced 29 clean gpt-5.4 trajectories
   used here.

2. **Convert** the bench's per-(model, scenario) results into
   training-format JSONL:

   ```bash
   uv run python data/seed/convert.py \
       --bench eval/results/benchmarks/<bench_id>/summary.json \
       --model gpt-5.4 \
       --out data/seed/v0/<name>.jsonl
   ```

3. **Review** the seeds. Reads REVIEW.md's hard rules + the
   token-level grounding-artifact heuristic, stamps `review_status`
   in place:

   ```bash
   uv run python data/seed/review.py data/seed/v0/<name>.jsonl
   ```

4. **Generate variants** (surface-detail substitutions for
   generalization):

   ```bash
   uv run python data/seed/vary.py \
       --in data/seed/v0/<name>.jsonl \
       --variants 10 \
       --out data/seed/varied/v0/<name>-varied.jsonl
   ```

5. **Synthesize negatives** (wrong call + K8sGPT-shape error +
   recovery):

   ```bash
   uv run python data/seed/synthesize_negatives.py \
       --in data/seed/v0/<name>.jsonl \
       --out data/seed/varied/v0/negatives-<date>.jsonl
   ```

6. **Snapshot K8sGPT tools/list** (one-time per K8sGPT version;
   spins up a throwaway kind cluster + k8sgpt serve):

   ```bash
   uv run python data/seed/snapshot_tools.py
   ```

   After this lands, re-run step 2 so the `tools` field is
   populated in the records.

## Uploading to Hugging Face

The current corpus (data/seed/v0/ + data/seed/varied/v0/) is ready
to publish. Manual recipe using `huggingface-cli`:

```bash
# One-time setup (not committed; uses HF_TOKEN from your shell env)
pip install --user huggingface_hub
huggingface-cli login   # paste a write-scoped token

# Create the dataset repo (one-time)
huggingface-cli repo create kubelm-seed-v0 --type dataset

# Pack and upload
cp data/seed/DATASET_CARD.md /tmp/README.md   # HF expects README.md
huggingface-cli upload \
    rbentaarit/kubelm-seed-v0 \
    /tmp/README.md \
    README.md \
    --repo-type dataset

huggingface-cli upload \
    rbentaarit/kubelm-seed-v0 \
    data/seed/v0/gpt-5.4-2026-05-12.jsonl \
    seeds/gpt-5.4-2026-05-12.jsonl \
    --repo-type dataset

huggingface-cli upload \
    rbentaarit/kubelm-seed-v0 \
    data/seed/varied/v0/gpt-5.4-2026-05-12-varied.jsonl \
    variants/gpt-5.4-2026-05-12-varied.jsonl \
    --repo-type dataset

huggingface-cli upload \
    rbentaarit/kubelm-seed-v0 \
    data/seed/varied/v0/negatives-2026-05-12.jsonl \
    negatives/negatives-2026-05-12.jsonl \
    --repo-type dataset
```

Or, with the `huggingface_hub` Python library (per-file calls are
the same set):

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj="data/seed/DATASET_CARD.md",
                path_in_repo="README.md",
                repo_id="rbentaarit/kubelm-seed-v0",
                repo_type="dataset")
# ... repeat for each JSONL
```

The upload is intentionally not scripted (no `upload_to_hf.py`)
because it's a one-shot publishing action; the maintainer should
review what's going up rather than run a script. When kubelm-pro
or kubelm-edge starts publishing, a packaged script is fair game.

## Current corpus stats

As of 2026-05-13:

- **29** reviewed seeds (all `accepted`, all `grounding_failed_v1_artifact: true`)
- **290** mechanical variants (10× per seed, surface-detail substituted)
- **46** synthetic negatives (17 wrong-resource-type + 29 hallucinated-tool-name)
- **365** total trajectories (~9.3 MB on disk)
- All trajectories pinned to K8sGPT `0.4.32`, MCP protocol `2025-03-26`

## Known limitations / followups

See the "Limitations" section in `DATASET_CARD.md` for the full
list. The most load-bearing:

1. Tools list is currently `null` in records (snapshot_tools.py
   hasn't been run for K8sGPT 0.4.32 yet).
2. Negative-example recovery prose is templated; a model trained
   on them may pick up exact phrasings. Hand-vary before scaling
   the corpus.
3. All positives come from gpt-5.4; no model-style diversity.
4. The v1 grounding analyzer is brittle to structured paraphrase
   (see PROJECT.md decisions log 2026-05-12 for the audit).
