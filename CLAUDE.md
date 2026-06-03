# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is at the **specification stage** — there is no implementation code yet. It contains only design documents. The deliverable is itself an **Agent/skill** (likely a Claude Code skill), not a conventional application: it reads a research-direction description and autonomously reproduces and iteratively optimizes open-source models.

When implementing, the project will not yet be a git repository (`git init` is needed first).

## What is being built

The spec is in [prd-20260521.md](prd-20260521.md) (Chinese). `autoexplore` is an autonomous agent that takes a *research-direction description* as input and runs two phases:

**Phase 1 — Baseline model selection:**
1. Clarify the research direction with the user.
2. Build a unified test set and design strongly-correlated evaluation metrics.
3. Find candidate open-source models; rank by officially-reported performance; pick ≤3.
4. Reproduce models by priority. Per-model reproduction loop:
   - Create a Docker environment (reuse if already built).
   - Download required packages and model weights inside Docker.
   - Run inference to verify the model works (use example code if provided, else write minimal test code).
   - On success exit the loop; on failure adjust the Docker environment and retry.
5. Run inference over the test set, compute metrics, select the best model for Phase 2.

**Phase 2 — Iterative optimization (loops indefinitely until manually stopped):**
1. Analyze metrics to identify the model's weaknesses.
2. Search trusted sources — **Arxiv, Hugging Face, paperswithcode** — for improvement directions; pick the 3 most promising.
3. Implement all 3; train if needed (use the paper's public dataset, or collect a similar public dataset).
4. Evaluate all 3 on the test set; keep the best metric as the new best.
5. Repeat.

## Compute environment

`autoexplore` targets a **multi-GPU / cloud** environment — larger scale than the `autoresearch` example. Carry over the example's *principles* (below) but **not** its specific single-GPU, fixed-5-minute-budget assumptions: experiments may use multiple GPUs, longer/variable time budgets, and Docker setups sized accordingly.

## Design template: the `autoresearch` example

[examples/autoresearch_program.md](examples/autoresearch_program.md) is the closest existing pattern for the autonomous experiment loop and should be treated as the reference design. Key principles to carry over:

- **A fixed, read-only evaluation harness is the ground truth metric.** In the example, `prepare.py` (data prep, tokenizer, evaluation) is never modified; only the experiment file (`train.py`) is touched. Mirror this separation: keep the test set and metric computation immutable so results are comparable across experiments.
- **Keep/discard/crash loop with branch advancement.** Each experiment: tune → commit → run with output redirected to a log (`> run.log 2>&1`, never flood context) → grep the key metric → if improved keep and advance the branch, else `git reset` back. Log every run to a `results.tsv` (tab-separated, kept *untracked*) with columns `commit  metric  memory_gb  status  description`.
- **Autonomy.** Once the loop starts, do not pause to ask whether to continue — run until manually stopped. If out of ideas, read referenced papers, re-read in-scope files, combine near-misses, or try more radical changes.
- **Bounded retries on crashes.** Fix trivial errors and re-run; abandon fundamentally broken ideas after a few attempts and log them as `crash`.
- **Simplicity criterion.** Prefer simpler solutions; a tiny metric gain that adds hacky complexity is not worth it, while equal results from *removing* code is a win.

## Git conventions

Full rules are in [GIT_CONVENTIONS.md](GIT_CONVENTIONS.md). Highlights that affect day-to-day work:

- **Branches:** `main` (deployable), `develop` (integration), `feature/*`, `bugfix/*` (off `develop`), `hotfix/*` (off `main`).
- **Commits:** Conventional Commits — `<type>(<scope>): <subject>`. Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`. Subject: imperative, lowercase, no trailing period, ≤50 chars.
- **Merge strategy:** squash-and-merge for `feature`/`bugfix`; merge commits for `hotfix` and `develop → main`.
- **Never rebase already-pushed commits** — use merge to integrate.
- Note: the autonomous experiment loop (per the `autoresearch` template) runs on its own dedicated branch and uses direct commits + `git reset`, which is a different workflow from the feature-branch process above. Use the dedicated-branch workflow for experiment runs and the `GIT_CONVENTIONS.md` flow for building the agent itself.

## Language

Spec and primary documentation are written in Chinese; respond and document in the language the user uses.
