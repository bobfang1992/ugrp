# UGRP UI

Simple Streamlit interface for exploring user profiles and recommendations.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run the app
streamlit run ui/profile_viewer.py
```

The app will open in your browser at `http://localhost:8501`

## Features

### 📊 Profile Stats
- Genre preferences (from liked movies)
- Decade distribution
- Recent liked movies

### 🎬 Rating History
- All user ratings with filtering
- Sort by timestamp, rating, title, or year
- Rating distribution chart

### ⭐ Recommendations
- Top-200 candidates from ALS model
- Adjustable top-N display
- Score distribution visualization

### 📈 Visualizations
- Rating history by movie year
- Rating vs. popularity scatter plot
- Genre coverage analysis

## Usage Tips

- **Quick picks**: Use sidebar buttons for Users 1, 100, or 1000
- **Filtering**: Multi-select ratings to focus on specific score ranges
- **Interactive charts**: Hover for details, click legends to toggle

## Requirements

Make sure you've trained the model first:

```bash
python src/ugrp/recsys/data_loader.py
python src/ugrp/recsys/model.py
python src/ugrp/profile/profile_builder.py
```

## Next Steps

This viewer shows the **base recommendations** before any control/reranking.

In M2, we'll add:
- Control JSON input
- Real-time reranking
- Explanation display
- Counterfactual comparison
