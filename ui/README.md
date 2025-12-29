# UGRP UI

Streamlit interfaces for exploring profiles and getting personalized recommendations.

## Apps

### 1. Profile Viewer - Explore Existing Users
```bash
streamlit run ui/profile_viewer.py
```
Explore the 6,040 users in the dataset with their profiles and recommendations.

### 2. My Profile - Create Your Own Profile
```bash
streamlit run ui/my_profile.py
```
Pick 10 movies you like and get 20 personalized recommendations!

---

Both apps will open in your browser at `http://localhost:8501`

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
