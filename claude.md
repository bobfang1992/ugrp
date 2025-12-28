# UGRP Next Steps

## Current Status
- ✅ Data downloaded (ML-1M and ML-20M in `data/raw/`)

## Next Steps (M1: Base + Profile)

### 1. Data Processing
- Load and parse MovieLens data
- Convert to parquet for faster access
- Extract movie metadata (title, year, genres)
- Basic EDA (distributions, sparsity, etc.)

### 2. Base Recommender
- Train simple ALS/MF model
- Generate Top-200 candidates per user
- Sanity check: compute Recall@10, NDCG@10

### 3. Profile Builder
- Aggregate user stats (top genres, year preferences, avg popularity)
- Create structured profile JSON
- Simple text summary (no LLM needed yet)

## Decision Needed
- Start with ML-1M or ML-20M? (recommend ML-1M for speed)

## After M1
- M2: Control JSON schema + deterministic reranker
- M3: LLM integration (parser + explainer)
- M4: ControlBench + evaluation
