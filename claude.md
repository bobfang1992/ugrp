# UGRP Progress Tracker

## ✅ M1 Complete: Base + Profile

### Completed
1. **Data Processing**
   - ✅ ML-1M: 3,883 movies, 1M ratings, 6,040 users
   - ✅ ML-20M: 27,278 movies, 20M ratings, 138,493 users
   - ✅ Converted to parquet (`data/processed/`)
   - ✅ Movie metadata extracted (title, year, genres, popularity)
   - ✅ IMDb/TMDB links added
   - ✅ EDA script (`scripts/eda.py`)

2. **Base Recommender**
   - ✅ ALS models trained (64 factors, 15 iterations)
   - ✅ ML-1M: Top-200 candidates per user (1.2M total)
   - ✅ ML-20M: Top-200 candidates per user (27.7M total)
   - ✅ Models saved (`als_model.pkl`, `als_model_20m.pkl`)

3. **Profile Builder**
   - ✅ ML-1M: 6,040 user profiles
   - ✅ ML-20M: 138,493 user profiles
   - ✅ Stats: top genres, year prefs, popularity bias, exploration score
   - ✅ Multiprocessing optimization (4-6x speedup for ML-20M)
   - ✅ Schema documented (`docs/profile_schema.md`)

4. **UI (Streamlit Multi-page App)**
   - ✅ Home page with system status
   - ✅ Profile Viewer (explore existing users)
   - ✅ My Profile (create custom profile, get recs)
   - ✅ Model Performance (evaluation metrics visualization)
   - ✅ Dataset selector (switch between ML-1M and ML-20M)

5. **Model Evaluation**
   - ✅ Train/test split (80/20 temporal per user)
   - ✅ Evaluation metrics: P@K, R@K, NDCG@K, HR@K, MAP@K
   - ✅ Evaluation module (`src/ugrp/eval/`)
   - ✅ Results saved to JSON, visualized in UI

### Key Files
- `src/ugrp/recsys/data_loader.py` - Data loading & cleaning, train/test split
- `src/ugrp/recsys/model.py` - ALS recommender training & evaluation
- `src/ugrp/profile/profile_builder.py` - User profiling
- `src/ugrp/recsys/movie_links.py` - IMDb/TMDB links
- `src/ugrp/eval/evaluator.py` - Evaluation metrics (P@K, NDCG@K, etc.)
- `ui/Home.py` - Landing page
- `ui/pages/1_Profile_Viewer.py` - Existing user profiles
- `ui/pages/2_My_Profile.py` - Custom profile creation
- `ui/pages/3_Model_Performance.py` - Evaluation metrics visualization
- `docs/profile_schema.md` - Profile JSON schema

### Training Commands
```bash
# Activate environment
source .venv/bin/activate

# ML-1M (smaller, faster)
python src/ugrp/recsys/data_loader.py          # Creates train/test split (80/20 temporal)
python src/ugrp/recsys/model.py                # Trains on train, evaluates on test
python src/ugrp/profile/profile_builder.py     # Builds user profiles (~10 sec)

# ML-20M (larger, more comprehensive)
python src/ugrp/recsys/data_loader.py --dataset ml-20m
python src/ugrp/recsys/model.py --dataset ml-20m
python src/ugrp/profile/profile_builder.py --dataset ml-20m  # Auto-parallel (~3-5 min)

# Run UI
streamlit run ui/Home.py
```

### Evaluation Metrics
Models are evaluated on temporal test set (20% most recent ratings per user):
- **Precision@K, Recall@K**: Relevance metrics
- **NDCG@K**: Ranking quality
- **Hit Rate@K, MAP@K**: User satisfaction metrics
- K values: 10, 20, 50

View results in UI: **Model Performance** page

---

## 🎯 Next: M2 - Control JSON Schema + Deterministic Reranker

### Goal
Build the control layer that takes user preferences as JSON and re-ranks candidates deterministically.

### Tasks
1. **Control JSON Schema** (`src/ugrp/control/`)
   - Define schema v0.1 (constraints, preferences, ui, meta)
   - JSON validator
   - Example controls

2. **Deterministic Reranker** (`src/ugrp/rerank/`)
   - Hard constraint filtering (genre, year)
   - Soft preference scoring (genre weights, novelty, popularity)
   - MMR-style diversity selection
   - Score breakdown (auditable)

3. **Evidence Builder** (`src/ugrp/explain/`)
   - Per-item explanation structure
   - Constraint pass/fail tracking
   - Component score breakdown

### After M2
- M3: LLM integration (intent parser + explanation renderer)
- M4: ControlBench + cross-LLM evaluation
