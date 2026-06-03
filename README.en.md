<div align="center">

# autoexplore

**An autonomous agent that reproduces and iteratively optimizes open-source models for a research direction**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg)](pyproject.toml)
[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-D97757.svg)](https://claude.com/claude-code)
[![Status](https://img.shields.io/badge/status-active-success.svg)](#project-status)

[简体中文](README.md) | English

</div>

---

## Introduction

**autoexplore** is an autonomous agent delivered as a set of [Claude Code](https://claude.com/claude-code) **Skills**. Give it a **research-direction description** and it runs the whole pipeline for you: *find candidate models → reproduce them in Docker → evaluate on a unified test set → pick the best baseline → keep optimizing* — and it can deploy the result as an interactive web app in one step.

It is not a conventional application but a set of **task- and model-agnostic, reusable skills** — the same pipeline works for visual question answering, segmentation, generation, or any open-source model direction with public weights.

## Key Features

- 🔭 **Direction clarification → frozen metrics**: aligns the research direction with you, then builds an **immutable** unified test set and evaluation metric so every model and experiment stays comparable.
- 🤖 **Automatic baseline reproduction**: searches Hugging Face / paperswithcode / arxiv for candidates, ranks them, and reproduces each in Docker (build/reuse image, download weights, verify inference, bounded retries), scores them on the test set, and selects the best.
- ♾️ **Never-ending optimization**: diagnoses model weaknesses → searches trusted sources (Arxiv / Hugging Face / paperswithcode) for improvement directions → implements and trains in parallel → scores → promotes a new backbone only on a relative **+5%** gain — looping until you stop it.
- 🌐 **One-step web deployment**: reuses the reproduced `infer.py` to spin up a **Gradio frontend + one backend container per model**, with lazy model loading and idle auto-unload.
- 🧰 **Deterministic script backbone**: GPU selection, Docker orchestration, inference, scoring, and the state machine are handled by tested Python scripts under `scripts/`. The agent only makes decisions — runs are interruptible and resumable.

## Workflow Overview

```
research-direction description
     │
     ▼
┌──────────────────────── Phase 1: Baseline selection ────────────────────────┐
│ 1. Clarify direction → 2. Build test set (frozen) → 3. Design metrics (frozen)│
│ 4. Search candidates → 5. Rank & pick ≤3 →                                    │
│ 6-9. Reproduce each in Docker + infer + score → 10. Pick the highest-scoring  │
└──────────────────────────────────────────────────────────────────────────────┘
     │  baseline model
     ▼
┌────────────────────── Phase 2: Iterative optimization (never-ending) ─────────┐
│ Diagnose weaknesses → inference-pipeline tuning gate (cheap first) →           │
│ LOOP: pick 3 directions → implement/train → score → promote on relative +5%    │
│ (never pauses to ask, until you manually stop it)                              │
└──────────────────────────────────────────────────────────────────────────────┘
     │  best model + infer.py
     ▼
   Web deployment (Gradio frontend + per-model backend containers)
```

## Environment

autoexplore targets a **multi-GPU cloud server**:

| Requirement | Notes |
|-------------|-------|
| Claude Code | Host runtime; it loads and orchestrates the skills |
| Python | ≥ 3.12 |
| [uv](https://docs.astral.sh/uv/) | Dependency & script runner (recommended) |
| Docker | Isolated environment for reproduction/deployment |
| NVIDIA Container Runtime | GPU access inside containers (`--runtime=nvidia`) |
| GPU | Multiple cards (reference config: 8), auto-selected by free memory |

## Installation

```bash
# 1. Clone
git clone <repo-url> autoexplore
cd autoexplore

# 2. Install dependencies (script runtime)
uv sync

# 3. Load the skills in Claude Code
#    The three skills under skills/ are ready to use — invoke them
#    on demand within a Claude Code session.
```

## Usage

In a Claude Code session, describe your research direction and invoke the skill for each phase:

| Skill | When to use |
|-------|-------------|
| **`autoexplore-phase1`** | Reproduce and evaluate ≤3 open-source models for a research direction; pick the highest-scoring baseline. |
| **`autoexplore-phase2`** | Run never-ending optimization on the phase-1 baseline; promote a new backbone on each relative +5% gain. |
| **`autoexplore-webdeploy`** | Deploy a reproduced/optimized `infer.py` as an interactive web app. |

> Each phase has a single **human checkpoint** (confirm direction, test set, and metrics). After that the agent runs autonomously and won't pause to ask step by step. Each run's working directory lives under `runs/<tag>/` (untracked by default).

### Deterministic scripts at a glance

The agent relies on a set of tested Python scripts for the critical deterministic steps:

| Script | Purpose |
|--------|---------|
| `scripts/gpu_select.py` | Select GPUs by free memory; print `CUDA_VISIBLE_DEVICES` |
| `scripts/docker_env.py` | Docker check / image build & reuse / container run |
| `scripts/run_inference.py` | In-container inference → `predictions.jsonl` |
| `scripts/compute_metrics.py` | Call the frozen `evaluate.py` → `metrics.json` |
| `scripts/progress.py` | Phase-1 progress persistence and result rollup |
| `scripts/phase2_state.py` | Phase-2 state machine / promotion gate / experiment dispatch |
| `scripts/diagnose.py` | Black-box `evaluate.py` calls for weakness breakdown |
| `scripts/directions_schema.py` | JSON-schema validation of optimization directions |
| `scripts/train_launch.py` | Data provenance / multi-GPU training launch / checkpoint resume |

## Design Disciplines

These constraints are why results are trustworthy and the pipeline is resumable:

- **Evaluation is ground truth and immutable**: once confirmed, the test set `dataset/` and the scoring script `evaluate.py` are frozen, keeping all models and experiments comparable.
- **Container discipline**: every `docker run` carries `--user $UID:$GID --runtime=nvidia`; model weights/data land in the in-repo `caches/{modelscope,huggingface,torch}` mounted `:ro` — never relying on `~/.cache`.
- **Bounded retries**: at most 3 retries per model/experiment; failures are marked `crash` without blocking siblings.
- **Interruptible & resumable**: entry points read the progress/state file first, skip finished items, and never roll back a promoted backbone.
- **Promotion only on a relative +5%** of the primary metric, to avoid backbone churn from random noise.
- **Logs don't pollute context**: container/training output is redirected to log files and only `tail`ed on failure.

## Repository Layout

```
autoexplore/
├── skills/                       # The three Claude Code skills (core deliverable)
│   ├── autoexplore-phase1/       #   Phase 1: reproduce baseline
│   ├── autoexplore-phase2/       #   Phase 2: iterative optimization
│   └── autoexplore-webdeploy/    #   Web deployment
├── scripts/                      # Deterministic Python scripts (GPU/Docker/infer/score/state)
├── tests/                        # pytest suite (GPU cases skipped by default)
├── docs/                         # Design specs and implementation plans
├── examples/                     # autoresearch reference design (autonomous-loop pattern)
├── 需求文档-20260521.md           # Original requirements doc (Chinese)
├── CLAUDE.md                     # Project guidance for Claude Code
├── GIT_CONVENTIONS.md            # Git branch & commit conventions
├── pyproject.toml
└── LICENSE
```

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests (cases needing real Docker + GPU are skipped by default)
uv run pytest

# Run all cases on a machine with Docker + GPU
uv run pytest -m gpu
```

## Contributing

Contributions are welcome! Please follow [GIT_CONVENTIONS.md](GIT_CONVENTIONS.md):

- **Branches**: `feature/*`, `bugfix/*` (off `develop`), `hotfix/*` (off `main`).
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) — `<type>(<scope>): <subject>`.
- **Merging**: squash-and-merge for `feature` / `bugfix`; **never rebase** already-pushed commits — integrate with merge.

## Project Status

Active — the three skills and the deterministic script backbone are implemented and test-covered; the core pipeline continues to evolve.

## License

Released under the [Apache License 2.0](LICENSE).

## Acknowledgements

- Built on the Skill mechanism of [Claude Code](https://claude.com/claude-code).
- The autonomous-experiment-loop pattern draws on [`examples/autoresearch_program.md`](examples/autoresearch_program.md).
