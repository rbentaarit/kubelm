# kubelm

**A small, Kubernetes-specialized language model that ships with your cluster.**

`kubelm` is an open project to build a family of small (1B–7B), domain-specialized language models for Kubernetes diagnostics and remediation — designed to run on the cluster itself, with no GPU and no external API calls, while matching or exceeding the K8s-task quality of much larger general-purpose local models.

> **Status:** Early. The benchmark and methodology come first. The models come second. This README describes where the project is heading and how to follow along.

---

## Why this exists

Tools like [K8sGPT](https://k8sgpt.ai) have made AI-assisted Kubernetes diagnostics genuinely useful, but the quality depends almost entirely on the language model behind them. Today, that means one of two trade-offs:

- **Send cluster data to a frontier cloud model** (OpenAI, Anthropic, Bedrock). Excellent quality, but data leaves the cluster, costs scale with usage, and air-gapped environments are excluded entirely.
- **Run a generic large local model** (Llama 3.3 70B, Qwen 2.5 32B). Good quality, but requires a GPU node pool — typically 40GB+ VRAM — which most clusters don't have provisioned for AI workloads.

There is a missing third option: **a small specialized model that runs on the cluster you already have**. That's what `kubelm` aims to be.

The hypothesis is that a small model concentrated on Kubernetes-specific knowledge — pod lifecycle, scheduler decisions, CNI behavior, common operator failure modes, kubectl output patterns — can match larger general models on K8s diagnostic tasks while running on a single CPU node in seconds.

If true, this enables a future where AI-assisted diagnostics ships as a default capability of K8s distributions, not as an add-on requiring GPU infrastructure or third-party API access.

---

## Project goals

1. **Publish a public Kubernetes diagnostics benchmark.** A reproducible eval harness with a curated set of K8s failure scenarios, scored against frontier cloud models, large local models, and small local models. The benchmark exists before any model is trained — it's the foundation everything else builds on.

2. **Build an open Kubernetes failure-pattern taxonomy.** A categorized inventory of real-world K8s failure modes, frequency-weighted by how often they occur in practice. Useful as a community reference even outside this project.

3. **Train and release a tiered family of specialized models.** Three sizes targeting different cluster resource profiles, all fine-tuned on the same curated K8s dataset, all measured against the same benchmark.

4. **Integrate with K8sGPT as a drop-in backend.** No fork. Use the existing OpenAI-compatible local-model interface so users can swap in `kubelm` with one config change.

5. **Ship as cluster-native infrastructure.** A Helm chart that deploys the model and inference engine alongside K8sGPT, with sensible defaults, sizing guidance, and no external dependencies.

---

## The model ladder (planned)

| Tier        | Size      | Target hardware             | Target latency  | Use case                          |
|-------------|-----------|-----------------------------|-----------------|-----------------------------------|
| `kubelm-edge`     | 1–1.5B    | 2-core CPU, 2GB RAM         | < 5 sec         | Edge clusters, dev, CI            |
| `kubelm-standard` | 3B        | 4-core CPU, 4GB RAM         | 10–20 sec       | Production default                |
| `kubelm-pro`      | 7–8B      | 8-core CPU or small GPU     | 15–30 sec (CPU) | Large clusters, regulated env.    |

These are targets, not guarantees. The benchmark will tell us where each tier actually lands. If a tier doesn't earn its keep, it gets cut.

---

## Methodology

Dataset and model quality are determined by methodology, not by training scale. The methodology this project commits to:

- **Eval-first.** The benchmark is built before any fine-tuning. Every dataset addition, every training run, every model release is measured against the same eval suite. No vibe-based progress claims.
- **Frequency-weighted dataset.** Training examples are weighted by real-world failure frequency (the Pareto principle applied honestly). Common patterns get heavy coverage. The long tail gets representative coverage. No uniform spread that over-invests in edge cases.
- **Pattern-based, not instance-based.** Examples teach failure *patterns*, not surface combinations. CrashLoopBackOff in a Deployment vs. a StatefulSet is the same pattern; the dataset doesn't duplicate it.
- **Reproducible training.** Dataset, training scripts, and hyperparameters are all public. Anyone can re-run the training and verify the results.
- **Open-weight releases.** All released models are Apache 2.0 with weights on Hugging Face. No closed weights, no custom licenses.

---

## What's in this repo (and what isn't, yet)

This repo will grow in stages. Each stage is a separately-useful artifact:

- [ ] **Phase 1: Failure-pattern taxonomy** — markdown reference document covering ~100 K8s failure patterns, frequency-weighted, with example symptoms and root causes. Useful even without any model.
- [ ] **Phase 2: Evaluation harness** — Python framework for running K8s diagnostic scenarios against any OpenAI-compatible model, with automated scoring.
- [ ] **Phase 3: Public benchmark results** — comparison of frontier cloud models, large local models, and small local models on the K8s benchmark. First blog post.
- [ ] **Phase 4: Seed training dataset** — 100–500 expert-curated K8s diagnostic examples, public on Hugging Face, with full provenance and review notes.
- [ ] **Phase 5: First fine-tuned model release** — `kubelm-standard` (3B) on Hugging Face, with reproducible training pipeline.
- [ ] **Phase 6: K8sGPT integration** — Helm chart deploying the model and inference engine as a K8sGPT backend.
- [ ] **Phase 7: Model ladder expansion** — `kubelm-edge` and `kubelm-pro` variants, evaluated against the same benchmark.

Items will be checked off as they land. Each phase is shipped publicly before the next begins.

---

## Non-goals

To keep scope honest:

- **Not a hosted SaaS.** This project ships open code, open weights, and open methodology. Anyone is free to build a hosted product on top; that's not the goal here.
- **Not a replacement for cloud frontier models.** For users with no privacy/cost/availability constraints, GPT-4-class models will likely remain higher quality on the long tail of K8s problems. `kubelm` targets the cases where running locally matters.
- **Not a fork of K8sGPT.** This project integrates with K8sGPT via the existing local-backend interface. K8sGPT remains the analyzer and orchestrator; `kubelm` is the language model component.
- **Not a research project.** The goal is shippable infrastructure, not novel ML techniques. The methodology uses well-established fine-tuning practices (QLoRA on small open base models). Innovation is in the dataset, evaluation, and deployment story, not the training algorithm.

---

## How to follow along

- **GitHub:** This repo will be the source of truth for code, datasets, and benchmark results.
- **Hugging Face:** Models and datasets will be published as they're ready.
- **Blog posts:** Major milestones (benchmark results, model releases, lessons learned) will be written up. Links will be added here as they're published.

---

## Contributing

The project isn't ready for code contributions yet — the foundation is still being laid. The most useful contribution today is **failure-pattern submissions**: real K8s failure scenarios you've encountered, with diagnosis and fix.

Once Phase 1 (the taxonomy) is published, there will be a clear contribution process for adding patterns, validating examples, and reviewing dataset additions.

---

## License

- **Code:** Apache 2.0
- **Models (when released):** Apache 2.0
- **Dataset (when released):** CC BY 4.0

---

## A note on framing

This project is built in public for two reasons. First, the artifacts (benchmark, taxonomy, methodology) are valuable independent of whether the models themselves succeed — and the community benefits from them being open. Second, building in public forces methodological honesty: every claim is reproducible, every result is comparable, and shortcuts get caught early.

If the core hypothesis turns out to be wrong — if a well-prompted Llama 3.3 70B with RAG over K8s docs already saturates the achievable quality — that's a publishable result too, and it changes the project rather than ending it. The eval harness and pattern taxonomy remain useful regardless.

The work happens at the intersection of three communities: K8s/CNCF, applied ML, and platform engineering. Discussions, corrections, and collaboration from any of these are welcome.
