# Training Guide - UGRP M1

Quick reference for training the base recommender and building user profiles.

## Prerequisites

```bash
# 1. Download MovieLens data (if not already done)
mkdir -p data/raw
cd data/raw

curl -L -o ml-1m.zip http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip

cd ../..

# 2. Activate virtual environment
source .venv/bin/activate
```

## Training Pipeline

### Step 1: Process Raw Data
```bash
python src/ugrp/recsys/data_loader.py
```

**Output:**
- `data/processed/movies.parquet` - Cleaned movie metadata with year, genres, popularity
- `data/processed/ratings.parquet` - User ratings
- `data/processed/users.parquet` - User demographics

**What it does:**
- Parses ML-1M `.dat` files
- Extracts year from movie titles
- Splits genres into lists
- Computes popularity quantiles

---

### Step 2: Train ALS Model
```bash
python src/ugrp/recsys/model.py
```

**Output:**
- `data/processed/als_model.pkl` - Trained ALS model (64 factors)
- `data/processed/candidates.parquet` - Top-200 candidates per user (1.2M rows)

**What it does:**
- Trains ALS on 1M ratings (6,040 users × 3,706 movies)
- Generates Top-200 recommendations per user
- Displays sample recommendations for user 1

**Expected runtime:** ~30 seconds

---

### Step 3: Build User Profiles
```bash
python src/ugrp/profile/profile_builder.py
```

**Output:**
- `data/processed/user_profiles.json` - 6,040 user profiles

**What it does:**
- Aggregates stats from rating history
- Computes genre preferences, year ranges, popularity bias
- Calculates exploration scores
- Shows sample profiles for users 1, 100, 1000

**Expected runtime:** ~10 seconds

---

## Exploratory Data Analysis

```bash
# View dataset statistics
python scripts/eda.py
```

**Shows:**
- Rating distribution (1-5)
- Year distribution by decade
- Top genres with percentages
- Popularity stats (most/least rated movies)
- User activity levels

---

## Verification

Check that all files were generated:

```bash
ls -lh data/processed/
```

**Expected files:**
```
movies.parquet          # ~500 KB
ratings.parquet         # ~24 MB
users.parquet           # ~200 KB
als_model.pkl           # ~20 MB
candidates.parquet      # ~50 MB
user_profiles.json      # ~15 MB
```

---

## Quick Test

Test a single user's recommendations:

```bash
python -c "
from ugrp.recsys.model import ALSRecommender
import pandas as pd

# Load model
model = ALSRecommender.load('data/processed/als_model.pkl')
movies = pd.read_parquet('data/processed/movies.parquet')

# Get recommendations for user 1
candidates = model.get_candidates(user_id=1, n=10)

print('Top 10 recommendations for user 1:')
for movie_id, score in candidates:
    movie = movies[movies['movieId'] == movie_id].iloc[0]
    print(f'{score:.3f} - {movie[\"title_clean\"]} ({movie[\"year\"]:.0f})')
"
```

---

## Re-training

To re-train from scratch:

```bash
# Delete processed data
rm -rf data/processed/*

# Re-run pipeline
python src/ugrp/recsys/data_loader.py
python src/ugrp/recsys/model.py
python src/ugrp/profile/profile_builder.py
```

---

## Troubleshooting

**Issue:** `FileNotFoundError: data/raw/ml-1m`
**Fix:** Download MovieLens data first (see Prerequisites)

**Issue:** `ModuleNotFoundError: No module named 'ugrp'`
**Fix:** Run `uv pip install -e .` from project root

**Issue:** Import errors from implicit
**Fix:** Reinstall dependencies: `uv pip install -e . --force-reinstall`

---

## Next Steps

After M1 training completes, proceed to M2:
- Control JSON schema definition
- Deterministic reranker implementation
- Evidence builder for explanations

See `claude.md` for M2 task breakdown.
