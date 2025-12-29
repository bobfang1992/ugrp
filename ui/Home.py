"""
UGRP - User-Governed Recommender Playground
Main landing page with app navigation.

Run with: streamlit run ui/Home.py
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="UGRP - Home",
    page_icon="🎬",
    layout="wide"
)

# Header
st.title("🎬 User-Governed Recommender Playground")
st.markdown("### Explore user profiles and get personalized movie recommendations")

st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Welcome to UGRP!

    This demo showcases **controllable, explainable movie recommendations**
    powered by collaborative filtering and user profiles.

    ### What's Built (M1 Complete ✅)

    - **Dual Dataset Support**: Switch between ML-1M and ML-20M
    - **Base Recommender**: ALS models trained on both datasets
    - **User Profiles**: Full profiles with preferences and behavioral metrics
    - **Candidates**: Top-200 recommendations per user

    ### Choose an App

    Use the sidebar to navigate between different views:
    """)

with col2:
    st.info("""
    **Datasets Available**

    **MovieLens 1M**
    - 6,040 users
    - 3,883 movies
    - 1M ratings

    **MovieLens 20M**
    - 138,493 users
    - 27,278 movies
    - 20M ratings
    """)

# App cards
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 👥 Profile Viewer

    **Explore existing users from the dataset**

    - Switch between ML-1M (6K users) and ML-20M (138K users)
    - View detailed user profiles
    - See rating history and preferences
    - Explore genre/decade distributions
    - Check ALS model recommendations

    📊 Perfect for understanding the dataset and user behavior patterns.
    """)

    st.markdown("👈 *Select 'Profile Viewer' from the sidebar*")

with col2:
    st.markdown("""
    ### ⭐ My Profile

    **Create your own profile and get recommendations**

    - Choose ML-1M (3.9K movies) or ML-20M (27K movies)
    - Search and pick 10 movies you like
    - Rate each movie 1-5 stars
    - Get 20 personalized recommendations
    - See which movies influenced each rec

    🎯 Perfect for testing the recommender with your own taste!
    """)

    st.markdown("👈 *Select 'My Profile' from the sidebar*")

# Footer
st.markdown("---")

st.markdown("""
### 📚 Documentation

- **Spec**: See `docs/UGRP_Spec_v0.1.md` for full project specification
- **Training Guide**: See `TRAINING.md` for data processing and model training
- **Profile Schema**: See `docs/profile_schema.md` for user profile JSON structure

### 🚀 Next Steps (M2)

- Control JSON schema definition
- Deterministic reranker with constraints
- Evidence builder for explanations
- LLM integration for intent parsing
""")

# Verify data
st.sidebar.markdown("---")
st.sidebar.header("System Status")

data_dir = Path("data/processed")

# Check ML-1M files
st.sidebar.markdown("**ML-1M (Required)**")
ml1m_files = {
    "Movies": "movies.parquet",
    "Ratings": "ratings.parquet",
    "ALS Model": "als_model.pkl",
    "Candidates": "candidates.parquet",
    "Profiles": "user_profiles.json"
}

ml1m_exists = True
for name, filename in ml1m_files.items():
    exists = (data_dir / filename).exists()
    ml1m_exists = ml1m_exists and exists
    st.sidebar.markdown(f"{'✅' if exists else '❌'} {name}")

# Check ML-20M files
st.sidebar.markdown("**ML-20M (Optional)**")
ml20m_files = {
    "Movies": "movies_20m.parquet",
    "Ratings": "ratings_20m.parquet",
    "ALS Model": "als_model_20m.pkl",
    "Candidates": "candidates_20m.parquet",
    "Profiles": "user_profiles_20m.json"
}

ml20m_exists = True
for name, filename in ml20m_files.items():
    exists = (data_dir / filename).exists()
    ml20m_exists = ml20m_exists and exists
    st.sidebar.markdown(f"{'✅' if exists else '❌'} {name}")

if not ml1m_exists:
    st.sidebar.warning("⚠️ ML-1M data missing. Run:")
    st.sidebar.code("""
python src/ugrp/recsys/data_loader.py
python src/ugrp/recsys/model.py
python src/ugrp/profile/profile_builder.py
    """, language="bash")
elif not ml20m_exists:
    st.sidebar.info("💡 To enable ML-20M, run:")
    st.sidebar.code("""
python src/ugrp/recsys/data_loader.py --dataset ml-20m
python src/ugrp/recsys/model.py --dataset ml-20m
python src/ugrp/profile/profile_builder.py --dataset ml-20m
    """, language="bash")
else:
    st.sidebar.success("✅ Both datasets ready!")
