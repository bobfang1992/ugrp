"""
User profile builder.
Aggregates user statistics from history to create structured profiles.

Following spec §12 M1: Profile stats + simple text summary (no LLM needed yet)
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import json


def convert_to_native_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {convert_to_native_types(k): convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class ProfileBuilder:
    """
    Builds user profiles from rating history.

    Profile includes:
    - Top genres (preference distribution)
    - Year preferences (decade distribution)
    - Popularity bias (tendency towards popular/niche items)
    - Rating behavior (avg rating, rating variance)
    - Exploration tendency (conservative/adventurous)
    """

    def __init__(self, movies_df, ratings_df):
        self.movies = movies_df
        self.ratings = ratings_df
        self.profiles = {}

    def build_profile(self, user_id):
        """
        Build profile for a single user.

        Returns:
            Dictionary with profile statistics
        """
        # Get user's rating history
        user_ratings = self.ratings[self.ratings['userId'] == user_id]

        if len(user_ratings) == 0:
            return None

        # Merge with movie metadata
        user_history = user_ratings.merge(
            self.movies[['movieId', 'title_clean', 'year', 'genres_list',
                         'num_ratings', 'popularity_quantile', 'avg_rating']],
            on='movieId'
        )

        # Only consider liked items (rating >= 4) for preference profiling
        liked_items = user_history[user_history['rating'] >= 4]

        profile = {
            'userId': user_id,
            'num_ratings': len(user_ratings),
            'avg_rating': float(user_ratings['rating'].mean()),
            'std_rating': float(user_ratings['rating'].std()),
        }

        # Genre preferences (from liked items)
        if len(liked_items) > 0:
            all_genres = []
            for genres in liked_items['genres_list']:
                all_genres.extend(genres)
            genre_counts = Counter(all_genres)
            total_genre_count = sum(genre_counts.values())

            # Top 5 genres with proportions
            top_genres = dict(genre_counts.most_common(5))
            profile['top_genres'] = {
                genre: count / total_genre_count
                for genre, count in top_genres.items()
            }
        else:
            profile['top_genres'] = {}

        # Year preferences (from liked items)
        if len(liked_items) > 0 and liked_items['year'].notna().any():
            profile['year_min'] = int(liked_items['year'].min())
            profile['year_max'] = int(liked_items['year'].max())
            profile['year_median'] = float(liked_items['year'].median())

            # Decade distribution
            decades = (liked_items['year'] // 10) * 10
            decade_counts = decades.value_counts()
            profile['top_decades'] = {
                int(decade): int(count)
                for decade, count in decade_counts.head(3).items()
            }
        else:
            profile['year_min'] = None
            profile['year_max'] = None
            profile['year_median'] = None
            profile['top_decades'] = {}

        # Popularity bias (-1 = niche lover, 0 = neutral, +1 = blockbuster lover)
        # Use all rated items for this metric
        if 'popularity_quantile' in user_history.columns:
            avg_popularity = user_history['popularity_quantile'].mean()
            # Map [0, 1] to [-1, +1] with 0.5 as neutral
            profile['popularity_bias'] = float((avg_popularity - 0.5) * 2)
        else:
            profile['popularity_bias'] = 0.0

        # Exploration tendency (based on rating diversity and niche consumption)
        # Conservative: high avg_popularity, low std_rating
        # Adventurous: low avg_popularity, high std_rating
        exploration_score = 0.0
        if len(user_ratings) >= 20:  # Need enough data
            # Component 1: Popularity (lower = more adventurous)
            pop_component = 1.0 - avg_popularity

            # Component 2: Rating variance (higher = more adventurous)
            rating_std = user_ratings['rating'].std()
            variance_component = min(rating_std / 1.5, 1.0)  # Normalize to [0, 1]

            # Component 3: Genre diversity (more genres = more adventurous)
            if len(liked_items) > 0:
                unique_genres = set()
                for genres in liked_items['genres_list']:
                    unique_genres.update(genres)
                genre_diversity = len(unique_genres) / 18  # ML-1M has 18 genres
            else:
                genre_diversity = 0

            exploration_score = (pop_component * 0.5 + variance_component * 0.3 + genre_diversity * 0.2)

        profile['exploration_score'] = float(exploration_score)

        # Categorize exploration tendency
        if exploration_score < 0.33:
            profile['exploration_tendency'] = 'conservative'
        elif exploration_score < 0.66:
            profile['exploration_tendency'] = 'medium'
        else:
            profile['exploration_tendency'] = 'adventurous'

        # Recent activity (last 10 liked movies)
        recent_liked = liked_items.sort_values('timestamp', ascending=False).head(10)
        profile['recent_liked'] = [
            {
                'movieId': int(row['movieId']),
                'title': row['title_clean'],
                'year': int(row['year']) if pd.notna(row['year']) else None,
                'rating': int(row['rating']),
                'genres': row['genres_list']
            }
            for _, row in recent_liked.iterrows()
        ]

        return profile

    def build_all_profiles(self):
        """
        Build profiles for all users in the dataset.

        Returns:
            Dictionary mapping userId -> profile
        """
        print(f"Building profiles for {len(self.ratings['userId'].unique())} users...")

        all_users = self.ratings['userId'].unique()
        for i, user_id in enumerate(all_users):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(all_users)} users...")

            profile = self.build_profile(user_id)
            if profile:
                self.profiles[user_id] = profile

        print(f"✓ Built {len(self.profiles)} profiles")
        return self.profiles

    def get_profile_summary(self, user_id):
        """
        Get a human-readable text summary of the profile.

        Args:
            user_id: User ID

        Returns:
            Text summary string
        """
        if user_id not in self.profiles:
            return f"No profile found for user {user_id}"

        profile = self.profiles[user_id]

        # Build summary
        lines = []
        lines.append(f"User {user_id} Profile")
        lines.append(f"  Activity: {profile['num_ratings']} ratings (avg: {profile['avg_rating']:.1f})")

        if profile['top_genres']:
            top_3_genres = list(profile['top_genres'].items())[:3]
            genre_str = ", ".join([f"{g} ({p*100:.0f}%)" for g, p in top_3_genres])
            lines.append(f"  Top genres: {genre_str}")

        if profile['year_median']:
            lines.append(f"  Year range: {profile['year_min']}–{profile['year_max']} (median: {profile['year_median']:.0f})")

        pop_bias = profile['popularity_bias']
        if pop_bias > 0.3:
            pop_desc = "prefers popular movies"
        elif pop_bias < -0.3:
            pop_desc = "prefers niche/indie movies"
        else:
            pop_desc = "balanced taste"
        lines.append(f"  Popularity: {pop_desc} (bias: {pop_bias:+.2f})")

        lines.append(f"  Exploration: {profile['exploration_tendency']} (score: {profile['exploration_score']:.2f})")

        return "\n".join(lines)

    def save_profiles(self, output_path="data/processed/user_profiles.json"):
        """Save all profiles to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert all numpy types to Python native types for JSON serialization
        profiles_json = convert_to_native_types(self.profiles)

        with open(output_path, 'w') as f:
            json.dump(profiles_json, f, indent=2)

        print(f"✓ Saved {len(self.profiles)} profiles to {output_path}")

    @classmethod
    def load_profiles(cls, path="data/processed/user_profiles.json"):
        """Load profiles from JSON"""
        with open(path, 'r') as f:
            profiles = json.load(f)

        # Convert string keys back to int
        profiles = {int(k): v for k, v in profiles.items()}

        print(f"✓ Loaded {len(profiles)} profiles from {path}")
        return profiles


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    movies = pd.read_parquet("data/processed/movies.parquet")
    ratings = pd.read_parquet("data/processed/ratings.parquet")

    # Build profiles
    builder = ProfileBuilder(movies, ratings)
    profiles = builder.build_all_profiles()

    # Save
    builder.save_profiles()

    # Show some examples
    print("\n=== Sample Profiles ===\n")
    for user_id in [1, 100, 1000]:
        print(builder.get_profile_summary(user_id))
        print()
