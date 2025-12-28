# UGRP Progress Tracker

## ✅ M1 Complete: Base + Profile

### Completed
1. **Data Processing**
   - ✅ ML-1M loaded and parsed (3,883 movies, 1M ratings, 6,040 users)
   - ✅ Converted to parquet (`data/processed/`)
   - ✅ Movie metadata extracted (title, year, genres, popularity)
   - ✅ EDA script (`scripts/eda.py`)

2. **Base Recommender**
   - ✅ ALS model trained (64 factors, 15 iterations)
   - ✅ Top-200 candidates generated per user (1.2M total)
   - ✅ Model saved (`data/processed/als_model.pkl`)
   - ✅ Candidates saved (`data/processed/candidates.parquet`)

3. **Profile Builder**
   - ✅ 6,040 user profiles built
   - ✅ Stats: top genres, year prefs, popularity bias, exploration score
   - ✅ Profiles saved (`data/processed/user_profiles.json`)
   - ✅ Schema documented (`docs/profile_schema.md`)

### Key Files
- `src/ugrp/recsys/data_loader.py` - Data loading & cleaning
- `src/ugrp/recsys/model.py` - ALS recommender
- `src/ugrp/profile/profile_builder.py` - User profiling
- `docs/profile_schema.md` - Profile JSON schema

### Training Commands
```bash
# Activate environment
source .venv/bin/activate

# 1. Process data (if needed)
python src/ugrp/recsys/data_loader.py

# 2. Train model + build profiles
python src/ugrp/recsys/model.py
python src/ugrp/profile/profile_builder.py

# 3. View stats
python scripts/eda.py
```

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
