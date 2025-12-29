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

### 4. Train models

```bash
# Process data and create train/test split
python src/ugrp/recsys/data_loader.py

# Train ALS model and evaluate
python src/ugrp/recsys/model.py

# Build user profiles
python src/ugrp/profile/profile_builder.py
```

### 5. Launch UI

```bash
streamlit run ui/Home.py
```

The UI provides:
- **Profile Viewer**: Explore existing users and their recommendations
- **My Profile**: Create custom profile and get personalized recommendations
- **Model Performance**: View evaluation metrics with interactive charts

## Project Structure

```
ugrp/
├── src/ugrp/          # main package
│   ├── profile/       # user profile builder
│   ├── recsys/        # base recommender (ALS/MF) + data loader
│   ├── eval/          # evaluation metrics (P@K, NDCG@K, etc.)
│   ├── control/       # control JSON schema (M2)
│   ├── rerank/        # deterministic reranker (M2)
│   ├── explain/       # evidence builder + LLM renderer (M2/M3)
│   ├── bench/         # ControlBench generator + evaluator (M4)
│   └── adapters/      # LLM adapters (M3/M4)
├── ui/                # Streamlit web interface
│   ├── Home.py        # landing page
│   └── pages/         # Profile Viewer, My Profile, Model Performance
├── data/              # datasets (gitignored)
├── docs/              # specs and documentation
└── outputs/           # logs and reports (gitignored)
```

## Documentation

- `docs/UGRP_Spec_v0.1.md` - full specification
- `docs/TRAINING.md` - training guide for base recommender + profiles
- `docs/profile_schema.md` - profile JSON schema and notes
