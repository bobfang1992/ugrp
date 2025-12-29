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
# ML-1M (smaller, faster)
python src/ugrp/recsys/data_loader.py

# ML-20M (larger dataset, optional)
python src/ugrp/recsys/data_loader.py --dataset ml-20m
```

**Output:**
- `data/processed/movies.parquet` - Cleaned movie metadata with year, genres, popularity
- `data/processed/ratings.parquet` - All user ratings
- `data/processed/train_ratings.parquet` - **Training set (80% per user, temporal)**
- `data/processed/test_ratings.parquet` - **Test set (20% per user, most recent)**
- `data/processed/users.parquet` - User demographics (ML-1M only)

**What it does:**
- Parses ML-1M `.dat` files or ML-20M `.csv` files
- Extracts year from movie titles
- Splits genres into lists
- Computes popularity quantiles
- **Creates temporal train/test split** (80/20 per user for evaluation)

---

### Step 2: Train ALS Model
```bash
# ML-1M
python src/ugrp/recsys/model.py

# ML-20M
python src/ugrp/recsys/model.py --dataset ml-20m
```

**Output:**
- `data/processed/als_model.pkl` - Trained ALS model (64 factors, 15 iterations)
- `data/processed/candidates.parquet` - Top-200 candidates per user
- `data/processed/evaluation.json` - **Evaluation metrics** (P@K, NDCG@K, HR@K, MAP@K)

**What it does:**
- **Trains ALS on training set only** (80% of ratings)
- **Evaluates on held-out test set** (20% of ratings)
- Generates Top-200 recommendations per user
- Computes evaluation metrics for K=10, 20, 50
- Displays sample recommendations for user 1

**Expected runtime:**
- ML-1M: ~1 minute (training + eval on 6K users)
- ML-20M: ~20 minutes (training + eval on 138K users)

**ML-1M Results:**
- NDCG@10: 0.1264, Precision@10: 11.3%, Hit Rate@10: 57.3%

**ML-20M Results:**
- NDCG@10: 0.1403, Precision@10: 12.3%, Hit Rate@10: 55.8%

---

### Step 3: Build User Profiles
```bash
# ML-1M (sequential - already fast)
python src/ugrp/profile/profile_builder.py

# ML-20M (auto-detect CPUs for parallel processing)
python src/ugrp/profile/profile_builder.py --dataset ml-20m

# ML-20M with specific number of workers
python src/ugrp/profile/profile_builder.py --dataset ml-20m --workers 8
```

**Output:**
- `data/processed/user_profiles.json` - User profiles for all users

**What it does:**
- Aggregates stats from rating history
- Computes genre preferences, year ranges, popularity bias
- Calculates exploration scores
- Shows sample profiles
- **Uses multiprocessing for ML-20M** (auto-detects CPU cores)

**Expected runtime:**
- ML-1M: ~10 seconds (6,040 users, sequential)
- ML-20M: ~26 minutes (138,493 users, parallel with 3 workers)
- ML-20M: ~45-50 minutes (sequential with --workers 1)

**Performance note:** Multiprocessing provides ~1.8x speedup (not linear) due to GIL contention in pandas operations. Using 3 workers balances performance vs resource usage.

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

**Expected files (ML-1M):**
```
movies.parquet          # ~500 KB
ratings.parquet         # ~24 MB
train_ratings.parquet   # ~19 MB (80%)
test_ratings.parquet    # ~5 MB (20%)
users.parquet           # ~200 KB
als_model.pkl           # ~20 MB
candidates.parquet      # ~50 MB
user_profiles.json      # ~15 MB
evaluation.json         # ~1 KB
```

**Additional files for ML-20M:**
```
movies_20m.parquet
ratings_20m.parquet
train_ratings_20m.parquet
test_ratings_20m.parquet
als_model_20m.pkl
candidates_20m.parquet
user_profiles_20m.json
evaluation_20m.json
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

## View Results in UI

Launch the Streamlit UI to explore profiles and evaluation metrics:

```bash
streamlit run ui/Home.py
```

**Available pages:**
- **Home**: Overview and system status
- **Profile Viewer**: Explore existing users (6K or 138K)
- **My Profile**: Create custom profile and get recommendations
- **Model Performance**: **View evaluation metrics** with interactive charts

**Model Performance page shows:**
- Precision@K, Recall@K, NDCG@K metrics for K=10, 20, 50
- Hit Rate@K, MAP@K statistics
- Side-by-side comparison of ML-1M vs ML-20M
- Interactive Plotly visualizations

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
