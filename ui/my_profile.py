"""
Streamlit app to create your own profile and get recommendations.
Run with: streamlit run ui/my_profile.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from ugrp.recsys.model import ALSRecommender

# Page config
st.set_page_config(
    page_title="UGRP - My Profile",
    page_icon="🎬",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    """Load movies data"""
    data_dir = Path("data/processed")
    movies = pd.read_parquet(data_dir / "movies.parquet")
    return movies

@st.cache_resource
def load_model():
    """Load trained ALS model"""
    return ALSRecommender.load("data/processed/als_model.pkl")

try:
    movies = load_data()
    model = load_model()

    st.title("🎬 Create Your Profile & Get Recommendations")
    st.markdown("Pick 10 movies you like, and we'll recommend 20 more based on your taste!")

    # Initialize session state
    if 'selected_movies' not in st.session_state:
        st.session_state.selected_movies = []
    if 'ratings' not in st.session_state:
        st.session_state.ratings = {}
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

    # Sidebar - Movie search and selection
    st.sidebar.header("1️⃣ Search & Add Movies")

    # Search box
    search_query = st.sidebar.text_input(
        "Search for a movie:",
        placeholder="e.g., Toy Story, Matrix, Titanic..."
    )

    # Filter movies based on search
    if search_query:
        # Search in title
        mask = movies['title'].str.contains(search_query, case=False, na=False)
        search_results = movies[mask].head(20)

        if len(search_results) > 0:
            st.sidebar.markdown(f"**Found {len(search_results)} movies:**")

            # Display search results
            for idx, row in search_results.iterrows():
                movie_id = row['movieId']

                # Skip if already selected
                if movie_id in st.session_state.selected_movies:
                    continue

                # Movie card
                with st.sidebar.container():
                    col1, col2 = st.sidebar.columns([3, 1])

                    with col1:
                        st.markdown(f"**{row['title_clean']}**")
                        st.caption(f"{row['year']:.0f} • {row['genres']}")
                        st.caption(f"⭐ {row['avg_rating']:.1f}/5 ({row['num_ratings']:.0f} ratings)")

                    with col2:
                        if st.button("➕", key=f"add_{movie_id}"):
                            if len(st.session_state.selected_movies) < 10:
                                st.session_state.selected_movies.append(movie_id)
                                st.session_state.ratings[movie_id] = 5  # Default to 5 stars
                                st.rerun()

                    st.sidebar.markdown("---")
        else:
            st.sidebar.info("No movies found. Try a different search term.")
    else:
        st.sidebar.info("👆 Type a movie name to search")

    # Main content
    st.header("2️⃣ Your Selected Movies")

    if len(st.session_state.selected_movies) == 0:
        st.info("Search and add movies from the sidebar to get started!")
    else:
        st.markdown(f"**Selected: {len(st.session_state.selected_movies)}/10 movies**")

        # Display selected movies
        selected_df = movies[movies['movieId'].isin(st.session_state.selected_movies)].copy()

        for idx, row in selected_df.iterrows():
            movie_id = row['movieId']

            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])

                with col1:
                    st.markdown(f"### {row['title_clean']}")
                    st.markdown(f"**{row['year']:.0f}** • {row['genres']}")
                    st.caption(f"⭐ Avg: {row['avg_rating']:.1f}/5 ({row['num_ratings']:.0f} ratings)")

                with col2:
                    # Rating slider
                    rating = st.slider(
                        "Your rating:",
                        1, 5, st.session_state.ratings[movie_id],
                        key=f"rating_{movie_id}"
                    )
                    st.session_state.ratings[movie_id] = rating

                with col3:
                    if st.button("🗑️ Remove", key=f"remove_{movie_id}"):
                        st.session_state.selected_movies.remove(movie_id)
                        del st.session_state.ratings[movie_id]
                        st.rerun()

                st.markdown("---")

        # Generate recommendations button
        if len(st.session_state.selected_movies) >= 5:
            st.header("3️⃣ Get Your Recommendations")

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("🎯 Generate Recommendations", type="primary", use_container_width=True):
                    # Generate recommendations
                    with st.spinner("Computing your personalized recommendations..."):
                        recs = generate_recommendations(
                            model, movies,
                            st.session_state.selected_movies,
                            st.session_state.ratings
                        )
                        st.session_state.recommendations = recs
                    st.success("✅ Recommendations ready!")
                    st.rerun()

            with col2:
                if st.button("🔄 Clear All", use_container_width=True):
                    st.session_state.selected_movies = []
                    st.session_state.ratings = {}
                    st.session_state.recommendations = None
                    st.rerun()

            # Show recommendations
            if st.session_state.recommendations is not None:
                st.header("🌟 Your Personalized Recommendations")

                recs_df = st.session_state.recommendations

                # Display recommendations
                for idx, row in recs_df.iterrows():
                    with st.container():
                        col1, col2 = st.columns([4, 1])

                        with col1:
                            st.markdown(f"### {idx + 1}. {row['title_clean']}")
                            st.markdown(f"**{row['year']:.0f}** • {row['genres']}")
                            st.caption(f"⭐ Avg: {row['avg_rating']:.1f}/5 ({row['num_ratings']:.0f} ratings)")

                        with col2:
                            st.metric("Match Score", f"{row['score']:.2f}")

                        # Show why this was recommended
                        st.caption(f"💡 Similar to: {row['similar_to']}")

                        st.markdown("---")

        else:
            st.info(f"👆 Add at least 5 movies to get recommendations (currently: {len(st.session_state.selected_movies)})")

except FileNotFoundError as e:
    st.error(f"Data not found: {e}")
    st.info("Make sure you've run the training pipeline first:")
    st.code("""
python src/ugrp/recsys/data_loader.py
python src/ugrp/recsys/model.py
    """)
except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)


def generate_recommendations(model, movies, selected_movie_ids, ratings, n=20):
    """
    Generate recommendations for a new user based on their selected movies.

    Uses item-item similarity from the ALS model's item factors.
    """
    # Get item factors from the model
    item_factors = model.model.item_factors  # shape: (n_items, n_factors)

    # Compute similarity scores for all movies
    scores = np.zeros(len(model.movie_id_map))

    for movie_id in selected_movie_ids:
        if movie_id not in model.movie_id_map:
            continue

        # Get internal index
        item_idx = model.movie_id_map[movie_id]

        # Get this item's factor vector
        item_vector = item_factors[item_idx]

        # Compute similarity to all other items (cosine similarity)
        similarities = item_factors @ item_vector

        # Weight by user's rating
        rating_weight = ratings[movie_id] / 5.0  # Normalize to [0, 1]
        scores += similarities * rating_weight

    # Exclude already selected movies
    for movie_id in selected_movie_ids:
        if movie_id in model.movie_id_map:
            item_idx = model.movie_id_map[movie_id]
            scores[item_idx] = -np.inf

    # Get top N
    top_indices = np.argsort(scores)[::-1][:n]

    # Map back to movie IDs
    recommendations = []
    for idx in top_indices:
        movie_id = model.reverse_movie_map[idx]
        score = float(scores[idx])

        # Find which input movie this is most similar to
        similar_to = find_most_similar(
            model, movie_id, selected_movie_ids, movies
        )

        recommendations.append({
            'movieId': movie_id,
            'score': score,
            'similar_to': similar_to
        })

    # Convert to DataFrame and merge with movie metadata
    recs_df = pd.DataFrame(recommendations)
    recs_df = recs_df.merge(
        movies[['movieId', 'title_clean', 'year', 'genres', 'avg_rating', 'num_ratings']],
        on='movieId'
    )

    return recs_df


def find_most_similar(model, rec_movie_id, selected_movie_ids, movies):
    """Find which selected movie this recommendation is most similar to"""
    if rec_movie_id not in model.movie_id_map:
        return "Unknown"

    rec_idx = model.movie_id_map[rec_movie_id]
    rec_vector = model.model.item_factors[rec_idx]

    max_sim = -1
    most_similar_id = None

    for movie_id in selected_movie_ids:
        if movie_id not in model.movie_id_map:
            continue

        item_idx = model.movie_id_map[movie_id]
        item_vector = model.model.item_factors[item_idx]

        similarity = np.dot(rec_vector, item_vector)

        if similarity > max_sim:
            max_sim = similarity
            most_similar_id = movie_id

    if most_similar_id:
        movie_info = movies[movies['movieId'] == most_similar_id].iloc[0]
        return f"{movie_info['title_clean']} ({movie_info['year']:.0f})"
    else:
        return "Unknown"
