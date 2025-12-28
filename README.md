# UGRP - User-Governed Recommender Playground

A reproducible demo + benchmark for user-controllable, faithful-explainable recommendations with cross-LLM evaluation.

## Setup

### 1. Install dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv pip install -e .
```

### 2. Download MovieLens datasets

```bash
# Create data directory
mkdir -p data/raw
cd data/raw

# Download ML-1M (1 million ratings)
curl -L -o ml-1m.zip http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip

# Download ML-20M (20 million ratings) - optional
curl -L -o ml-20m.zip http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
```

**Citation**: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>

### 3. Activate environment

```bash
source .venv/bin/activate
```

## Project Structure

```
ugrp/
├── src/ugrp/          # main package
│   ├── profile/       # user profile builder
│   ├── recsys/        # base recommender (ALS/MF)
│   ├── control/       # control JSON schema
│   ├── rerank/        # deterministic reranker
│   ├── explain/       # evidence builder + LLM renderer
│   ├── bench/         # ControlBench generator + evaluator
│   └── adapters/      # LLM adapters (GPT, Gemini, etc.)
├── data/              # datasets (gitignored)
├── docs/              # specs and documentation
└── outputs/           # logs and reports (gitignored)
```

## Documentation

See `docs/UGRP_Spec_v0.1.md` for the full specification.
