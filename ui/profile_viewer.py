"""
Streamlit app to explore UGRP user profiles and recommendations.
Run with: streamlit run ui/profile_viewer.py
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="UGRP Profile Viewer",
    page_icon="🎬",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    """Load all processed data"""
    data_dir = Path("data/processed")

    movies = pd.read_parquet(data_dir / "movies.parquet")
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    candidates = pd.read_parquet(data_dir / "candidates.parquet")

    with open(data_dir / "user_profiles.json") as f:
        profiles = json.load(f)

    return movies, ratings, candidates, profiles

try:
    movies, ratings, candidates, profiles = load_data()

    # Convert profile keys to int
    profiles = {int(k): v for k, v in profiles.items()}

    st.title("🎬 UGRP Profile Viewer")
    st.markdown("Explore user profiles, viewing history, and recommendations from the base ALS model")

    # Sidebar - User selection
    st.sidebar.header("Select User")
    user_ids = sorted(profiles.keys())

    # Quick picks
    st.sidebar.markdown("**Quick picks:**")
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("User 1"):
        selected_user = 1
    elif col2.button("User 100"):
        selected_user = 100
    elif col3.button("User 1000"):
        selected_user = 1000
    else:
        selected_user = st.sidebar.selectbox(
            "Or choose any user:",
            options=user_ids,
            index=0
        )

    profile = profiles[selected_user]
    user_ratings = ratings[ratings['userId'] == selected_user].copy()
    user_candidates = candidates[candidates['userId'] == selected_user].copy()

    # Merge with movie data
    user_ratings = user_ratings.merge(
        movies[['movieId', 'title_clean', 'year', 'genres', 'avg_rating', 'num_ratings']],
        on='movieId'
    )
    user_candidates = user_candidates.merge(
        movies[['movieId', 'title_clean', 'year', 'genres', 'avg_rating', 'num_ratings']],
        on='movieId'
    )

    # Main content
    st.header(f"👤 User {selected_user} Profile")

    # Profile card
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Ratings", profile['num_ratings'])
        st.metric("Avg Rating", f"{profile['avg_rating']:.2f}")

    with col2:
        pop_bias = profile['popularity_bias']
        pop_label = "Popular" if pop_bias > 0.3 else "Niche" if pop_bias < -0.3 else "Balanced"
        st.metric("Popularity Preference", pop_label)
        st.caption(f"Bias: {pop_bias:+.2f}")

    with col3:
        st.metric("Exploration Style", profile['exploration_tendency'].title())
        st.caption(f"Score: {profile['exploration_score']:.2f}")

    with col4:
        if profile['year_median']:
            st.metric("Year Preference", f"{profile['year_median']:.0f}")
            st.caption(f"Range: {profile['year_min']}-{profile['year_max']}")
        else:
            st.metric("Year Preference", "N/A")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Profile Stats",
        "🎬 Rating History",
        "⭐ Recommendations",
        "📈 Visualizations"
    ])

    # Tab 1: Profile Stats
    with tab1:
        st.subheader("Genre Preferences")
        if profile['top_genres']:
            genre_df = pd.DataFrame([
                {'Genre': g, 'Proportion': p}
                for g, p in profile['top_genres'].items()
            ])

            fig = px.bar(
                genre_df,
                x='Genre',
                y='Proportion',
                title='Top Genres (from liked movies, rating ≥ 4)',
                color='Proportion',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No genre preferences (no liked items)")

        st.subheader("Decade Distribution")
        if profile['top_decades']:
            decade_df = pd.DataFrame([
                {'Decade': str(d), 'Count': c}
                for d, c in sorted(profile['top_decades'].items())
            ])

            fig = px.bar(
                decade_df,
                x='Decade',
                y='Count',
                title='Favorite Decades (from liked movies)',
                color='Count',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No decade preferences available")

        st.subheader("Recent Liked Movies")
        if profile['recent_liked']:
            recent_df = pd.DataFrame(profile['recent_liked'])
            recent_df['genres_str'] = recent_df['genres'].apply(lambda x: ', '.join(x) if x else '')
            st.dataframe(
                recent_df[['title', 'year', 'rating', 'genres_str']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No recent liked movies")

    # Tab 2: Rating History
    with tab2:
        st.subheader(f"All {len(user_ratings)} Ratings")

        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            rating_filter = st.multiselect(
                "Filter by rating:",
                options=[1, 2, 3, 4, 5],
                default=[1, 2, 3, 4, 5]
            )
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                options=['timestamp', 'rating', 'title_clean', 'year'],
                index=0
            )

        filtered_ratings = user_ratings[user_ratings['rating'].isin(rating_filter)]
        filtered_ratings = filtered_ratings.sort_values(sort_by, ascending=False)

        # Display
        st.dataframe(
            filtered_ratings[[
                'title_clean', 'year', 'rating', 'genres',
                'avg_rating', 'num_ratings'
            ]].rename(columns={
                'title_clean': 'Title',
                'year': 'Year',
                'rating': 'User Rating',
                'genres': 'Genres',
                'avg_rating': 'Avg Rating',
                'num_ratings': '# Ratings'
            }),
            use_container_width=True,
            hide_index=True
        )

        # Rating distribution
        st.subheader("Rating Distribution")
        rating_counts = user_ratings['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={'x': 'Rating', 'y': 'Count'},
            title=f'User {selected_user} Rating Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Recommendations
    with tab3:
        st.subheader("Top Recommendations from ALS Model")
        st.caption("These are the Top-200 candidates before any control/reranking is applied")

        # Show top N
        top_n = st.slider("Show top N recommendations:", 10, 200, 20, 10)

        top_candidates = user_candidates.head(top_n)

        # Display
        st.dataframe(
            top_candidates[[
                'rank', 'title_clean', 'year', 'genres', 'score',
                'avg_rating', 'num_ratings'
            ]].rename(columns={
                'rank': 'Rank',
                'title_clean': 'Title',
                'year': 'Year',
                'genres': 'Genres',
                'score': 'ALS Score',
                'avg_rating': 'Avg Rating',
                'num_ratings': '# Ratings'
            }),
            use_container_width=True,
            hide_index=True
        )

        # Score distribution
        st.subheader("Score Distribution")
        fig = px.histogram(
            user_candidates,
            x='score',
            nbins=50,
            title='ALS Score Distribution (all 200 candidates)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 4: Visualizations
    with tab4:
        st.subheader("User Behavior Analysis")

        # Year distribution
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Rating History by Year**")
            year_dist = user_ratings.groupby('year').size().reset_index(name='count')
            fig = px.line(
                year_dist,
                x='year',
                y='count',
                title='Movies Rated by Release Year'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Rating vs. Popularity**")
            fig = px.scatter(
                user_ratings.sample(min(100, len(user_ratings))),
                x='num_ratings',
                y='rating',
                hover_data=['title_clean', 'year'],
                title='User Rating vs. Movie Popularity',
                labels={'num_ratings': 'Movie Popularity (# ratings)', 'rating': 'User Rating'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Genre analysis
        st.markdown("**Genre Coverage**")
        all_genres = []
        for genres_list in user_ratings['genres'].str.split('|'):
            all_genres.extend(genres_list)

        genre_counts = pd.Series(all_genres).value_counts().head(10)
        fig = px.bar(
            x=genre_counts.index,
            y=genre_counts.values,
            labels={'x': 'Genre', 'y': 'Count'},
            title='Top 10 Genres in Rating History'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Footer stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset Stats**")
    st.sidebar.metric("Total Users", len(profiles))
    st.sidebar.metric("Total Movies", len(movies))
    st.sidebar.metric("Total Ratings", len(ratings))

except FileNotFoundError as e:
    st.error(f"Data not found: {e}")
    st.info("Make sure you've run the training pipeline first:")
    st.code("""
python src/ugrp/recsys/data_loader.py
python src/ugrp/recsys/model.py
python src/ugrp/profile/profile_builder.py
    """)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.exception(e)
