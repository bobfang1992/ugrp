# UGRP UI

Multi-page Streamlit app for exploring profiles and getting personalized recommendations.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run the app (launches all pages)
streamlit run ui/Home.py
```

The app will open at `http://localhost:8501` with navigation in the sidebar.

## Pages

### 🏠 Home
Landing page with overview and system status check.

### 👥 Profile Viewer
Explore users from ML-1M (6,040 users) or ML-20M (138,493 users) with their profiles and recommendations.

### ⭐ My Profile
Pick 10 movies you like and get 20 personalized recommendations! Choose between ML-1M (3.9K movies) or ML-20M (27K movies).

### 📊 Model Performance
View evaluation metrics with interactive visualizations. Compare ML-1M vs ML-20M performance across Precision@K, NDCG@K, Hit Rate@K, and more.

---

## Profile Viewer Features

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

---

## My Profile Features

### 🔍 Movie Search
- Type any movie name to search
- Live results with movie details (year, genres, ratings)
- Add up to 10 movies to your profile

### ⭐ Rating System
- Rate each selected movie 1-5 stars
- Adjust ratings with sliders
- Remove movies you don't want

### 🎯 Recommendations
- Get 20 personalized recommendations
- Based on item-item similarity from ALS model
- Shows which movie each recommendation is similar to
- Match scores for each recommendation

---

## Usage Tips

- **User selection**: Use the dropdown in the sidebar to select any user
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
