"""
Streamlit app to create your own profile and get recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from ugrp.recsys.model import ALSRecommender
from ugrp.recsys.movie_links import add_links_to_movies

# Page config
st.set_page_config(
    page_title="UGRP - My Profile",
    page_icon="⭐",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    """Load movies data with links"""
    data_dir = Path("data/processed")
    movies = pd.read_parquet(data_dir / "movies.parquet")
    # Add IMDb/TMDB links
    movies = add_links_to_movies(movies)
    return movies

@st.cache_resource
def load_model():
    """Load trained ALS model"""
    return ALSRecommender.load("data/processed/als_model.pkl")

def render_movie_card(row, key_prefix, show_add_button=False, show_remove_button=False):
    """Render a movie card with details and links"""
    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(f"**{row['title_clean']}** ({row['year']:.0f})")
            st.caption(f"🎭 {row['genres']}")
            st.caption(f"⭐ {row['avg_rating']:.1f}/5 • 👥 {row['num_ratings']:.0f} ratings")

            # Links
            links = []
            if pd.notna(row.get('imdb_url')):
                links.append(f"[IMDb]({row['imdb_url']})")
            if pd.notna(row.get('tmdb_url')):
                links.append(f"[TMDB]({row['tmdb_url']})")
            if pd.notna(row.get('imdb_search_url')):
                links.append(f"[Search IMDb]({row['imdb_search_url']})")

            if links:
                st.markdown(" • ".join(links))

        with col2:
            if show_add_button:
                return st.button("➕ Add", key=f"{key_prefix}_add")
            elif show_remove_button:
                return st.button("🗑️", key=f"{key_prefix}_remove")

        st.markdown("---")
        return False


try:
    movies = load_data()
    model = load_model()

    st.title("⭐ Create Your Profile")
    st.markdown("Pick 10 movies you like, rate them, and get 20 personalized recommendations!")

    # Initialize session state
    if 'selected_movies' not in st.session_state:
        st.session_state.selected_movies = []
    if 'ratings' not in st.session_state:
        st.session_state.ratings = {}
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'show_browse' not in st.session_state:
        st.session_state.show_browse = False

    # Two columns layout
    col_search, col_selected = st.columns([1, 1])

    # LEFT COLUMN: Search & Browse
    with col_search:
        st.header("🔍 Find Movies")

        # Search box
        search_query = st.text_input(
            "Search by movie title:",
            placeholder="Type to search (e.g., Matrix, Toy Story, Titanic)...",
            help="Start typing to see matching movies"
        )

        # Browse all movies toggle
        if st.button("📚 Browse All Movies" if not st.session_state.show_browse else "🔍 Back to Search"):
            st.session_state.show_browse = not st.session_state.show_browse
            st.rerun()

        st.markdown("---")

        # Display search results or browse
        if st.session_state.show_browse:
            st.subheader("All Movies")

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                genre_filter = st.selectbox(
                    "Filter by genre:",
                    options=['All'] + sorted(
                        set(genre for genres in movies['genres'].str.split('|')
                            for genre in genres)
                    )
                )

            with col2:
                year_min = int(movies['year'].min())
                year_max = int(movies['year'].max())
                year_range = st.slider(
                    "Year range:",
                    year_min, year_max,
                    (year_min, year_max)
                )

            # Apply filters
            filtered = movies.copy()
            if genre_filter != 'All':
                filtered = filtered[filtered['genres'].str.contains(genre_filter, na=False)]
            filtered = filtered[(filtered['year'] >= year_range[0]) & (filtered['year'] <= year_range[1])]

            # Sort options
            sort_by = st.selectbox(
                "Sort by:",
                options=['Popularity (highest)', 'Rating (highest)', 'Title (A-Z)', 'Year (newest)']
            )

            if sort_by == 'Popularity (highest)':
                filtered = filtered.sort_values('num_ratings', ascending=False)
            elif sort_by == 'Rating (highest)':
                filtered = filtered.sort_values('avg_rating', ascending=False)
            elif sort_by == 'Title (A-Z)':
                filtered = filtered.sort_values('title_clean')
            elif sort_by == 'Year (newest)':
                filtered = filtered.sort_values('year', ascending=False)

            st.caption(f"Showing {len(filtered)} movies")

            # Pagination
            page_size = 10
            total_pages = (len(filtered) - 1) // page_size + 1
            page = st.number_input("Page:", 1, total_pages, 1) - 1

            start_idx = page * page_size
            end_idx = start_idx + page_size
            page_results = filtered.iloc[start_idx:end_idx]

            # Display results
            for idx, row in page_results.iterrows():
                if row['movieId'] in st.session_state.selected_movies:
                    continue

                if render_movie_card(row, f"browse_{row['movieId']}", show_add_button=True):
                    if len(st.session_state.selected_movies) < 10:
                        st.session_state.selected_movies.append(row['movieId'])
                        st.session_state.ratings[row['movieId']] = 5
                        st.rerun()
                    else:
                        st.warning("You've already selected 10 movies!")

        elif search_query:
            st.subheader("Search Results")

            # Filter movies based on search
            mask = movies['title'].str.contains(search_query, case=False, na=False)
            search_results = movies[mask].sort_values('num_ratings', ascending=False).head(20)

            if len(search_results) > 0:
                st.caption(f"Found {len(search_results)} movies")

                # Display search results
                for idx, row in search_results.iterrows():
                    if row['movieId'] in st.session_state.selected_movies:
                        continue

                    if render_movie_card(row, f"search_{row['movieId']}", show_add_button=True):
                        if len(st.session_state.selected_movies) < 10:
                            st.session_state.selected_movies.append(row['movieId'])
                            st.session_state.ratings[row['movieId']] = 5
                            st.rerun()
                        else:
                            st.warning("You've already selected 10 movies!")
            else:
                st.info("No movies found. Try a different search term.")
        else:
            st.info("👆 Type a movie name or browse all movies")

    # RIGHT COLUMN: Selected movies
    with col_selected:
        st.header("✨ Your Movies")
        st.caption(f"{len(st.session_state.selected_movies)}/10 selected")

        if len(st.session_state.selected_movies) == 0:
            st.info("Search and add movies to get started!")
        else:
            selected_df = movies[movies['movieId'].isin(st.session_state.selected_movies)].copy()

            for idx, row in selected_df.iterrows():
                movie_id = row['movieId']

                with st.container():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**{row['title_clean']}** ({row['year']:.0f})")
                        st.caption(f"{row['genres']}")

                        # Rating slider
                        rating = st.slider(
                            "Your rating:",
                            1, 5,
                            st.session_state.ratings[movie_id],
                            key=f"rating_{movie_id}",
                            label_visibility="collapsed"
                        )
                        st.session_state.ratings[movie_id] = rating

                    with col2:
                        st.markdown("⭐" * rating)
                        if st.button("🗑️", key=f"remove_{movie_id}"):
                            st.session_state.selected_movies.remove(movie_id)
                            del st.session_state.ratings[movie_id]
                            st.rerun()

                    st.markdown("---")

    # Recommendations section
    st.markdown("---")

    if len(st.session_state.selected_movies) >= 5:
        st.header("🎯 Get Your Recommendations")

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("✨ Generate Recommendations", type="primary", use_container_width=True):
                with st.spinner("Computing your personalized recommendations..."):
                    recs = generate_recommendations(
                        model, movies,
                        st.session_state.selected_movies,
                        st.session_state.ratings
                    )
                    st.session_state.recommendations = recs
                st.success("✅ Done!")
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

            for idx, row in recs_df.iterrows():
                with st.container():
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(f"### {idx + 1}. {row['title_clean']} ({row['year']:.0f})")
                        st.markdown(f"🎭 {row['genres']}")
                        st.caption(f"⭐ {row['avg_rating']:.1f}/5 • 👥 {row['num_ratings']:.0f} ratings")

                        # Links
                        links = []
                        if pd.notna(row.get('imdb_url')):
                            links.append(f"[IMDb]({row['imdb_url']})")
                        if pd.notna(row.get('tmdb_url')):
                            links.append(f"[TMDB]({row['tmdb_url']})")
                        if pd.notna(row.get('imdb_search_url')):
                            links.append(f"[Search IMDb]({row['imdb_search_url']})")
                        if links:
                            st.markdown(" • ".join(links))

                        st.caption(f"💡 Similar to: {row['similar_to']}")

                    with col2:
                        st.metric("Match", f"{row['score']:.2f}")

                    st.markdown("---")

    else:
        st.info(f"👆 Add at least 5 movies to get recommendations (currently: {len(st.session_state.selected_movies)})")

except FileNotFoundError as e:
    st.error(f"Data not found: {e}")
    st.info("Make sure you've run the training pipeline first.")
except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)


def generate_recommendations(model, movies, selected_movie_ids, ratings, n=20):
    """Generate recommendations using item-item similarity"""
    item_factors = model.model.item_factors
    scores = np.zeros(len(model.movie_id_map))

    for movie_id in selected_movie_ids:
        if movie_id not in model.movie_id_map:
            continue

        item_idx = model.movie_id_map[movie_id]
        item_vector = item_factors[item_idx]
        similarities = item_factors @ item_vector
        rating_weight = ratings[movie_id] / 5.0
        scores += similarities * rating_weight

    # Exclude selected movies
    for movie_id in selected_movie_ids:
        if movie_id in model.movie_id_map:
            item_idx = model.movie_id_map[movie_id]
            scores[item_idx] = -np.inf

    # Get top N
    top_indices = np.argsort(scores)[::-1][:n]

    recommendations = []
    for idx in top_indices:
        movie_id = model.reverse_movie_map[idx]
        score = float(scores[idx])
        similar_to = find_most_similar(model, movie_id, selected_movie_ids, movies)

        recommendations.append({
            'movieId': movie_id,
            'score': score,
            'similar_to': similar_to
        })

    recs_df = pd.DataFrame(recommendations)
    recs_df = recs_df.merge(
        movies[['movieId', 'title_clean', 'year', 'genres', 'avg_rating',
                'num_ratings', 'imdb_url', 'tmdb_url', 'imdb_search_url']],
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
