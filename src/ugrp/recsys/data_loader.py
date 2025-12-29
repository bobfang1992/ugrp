"""
Data loader for MovieLens datasets.
Loads ML-1M data and converts to pandas DataFrames.
"""

import pandas as pd
import re
from pathlib import Path


def load_ml1m(data_dir: str = "data/raw/ml-1m"):
    """
    Load MovieLens 1M dataset.

    Args:
        data_dir: Path to ML-1M directory

    Returns:
        movies_df, ratings_df, users_df
    """
    data_path = Path(data_dir)

    # Load movies: movieId::title::genres
    movies_df = pd.read_csv(
        data_path / "movies.dat",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1"
    )

    # Load ratings: userId::movieId::rating::timestamp
    ratings_df = pd.read_csv(
        data_path / "ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )

    # Load users: userId::gender::age::occupation::zipcode
    users_df = pd.read_csv(
        data_path / "users.dat",
        sep="::",
        engine="python",
        names=["userId", "gender", "age", "occupation", "zipcode"]
    )

    return movies_df, ratings_df, users_df


def extract_year(title: str) -> int:
    """Extract year from movie title like 'Toy Story (1995)'"""
    match = re.search(r'\((\d{4})\)$', title.strip())
    if match:
        return int(match.group(1))
    return None


def clean_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean movies dataframe:
    - Extract year from title
    - Split genres into list
    - Clean title (remove year suffix)
    """
    df = movies_df.copy()

    # Extract year
    df['year'] = df['title'].apply(extract_year)

    # Clean title (remove year)
    df['title_clean'] = df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)

    # Split genres
    df['genres_list'] = df['genres'].str.split('|')

    # Count number of genres per movie
    df['num_genres'] = df['genres_list'].apply(len)

    return df


def compute_movie_stats(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute movie-level statistics:
    - Number of ratings
    - Average rating
    - Rating std
    - Popularity quantile
    """
    movie_stats = ratings_df.groupby('movieId').agg(
        num_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean'),
        std_rating=('rating', 'std')
    ).reset_index()

    # Compute popularity quantile (0 = least popular, 1 = most popular)
    movie_stats['popularity_quantile'] = movie_stats['num_ratings'].rank(pct=True)

    return movie_stats


def prepare_ml1m_dataset(data_dir: str = "data/raw/ml-1m",
                         output_dir: str = "data/processed"):
    """
    Load, clean, and save ML-1M dataset to parquet.

    Returns:
        Dictionary with dataframes: movies, ratings, users, movie_stats
    """
    print("Loading ML-1M data...")
    movies_df, ratings_df, users_df = load_ml1m(data_dir)

    print(f"Loaded {len(movies_df)} movies, {len(ratings_df)} ratings, {len(users_df)} users")

    print("Cleaning movies data...")
    movies_df = clean_movies(movies_df)

    print("Computing movie statistics...")
    movie_stats = compute_movie_stats(ratings_df)

    # Merge stats back into movies
    movies_df = movies_df.merge(movie_stats, on='movieId', how='left')

    # Save to parquet
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_dir}...")
    movies_df.to_parquet(output_path / "movies.parquet", index=False)
    ratings_df.to_parquet(output_path / "ratings.parquet", index=False)
    users_df.to_parquet(output_path / "users.parquet", index=False)

    print("✓ Data processing complete!")

    return {
        'movies': movies_df,
        'ratings': ratings_df,
        'users': users_df,
        'movie_stats': movie_stats
    }


if __name__ == "__main__":
    import sys

    # Check for dataset argument
    dataset = "ml-1m"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--dataset":
            dataset = sys.argv[2] if len(sys.argv) > 2 else "ml-1m"

    print(f"Processing {dataset.upper()}...")

    if dataset == "ml-20m":
        # For ML-20M, use similar logic but with CSV format
        data_dir = "data/raw/ml-20m"
        output_dir = "data/processed"

        print("Loading ML-20M data...")
        movies = pd.read_csv(f"{data_dir}/movies.csv")
        ratings = pd.read_csv(f"{data_dir}/ratings.csv")

        print(f"Loaded {len(movies)} movies, {len(ratings)} ratings")

        # Clean movies (same logic as ML-1M)
        movies['year'] = movies['title'].apply(extract_year)
        movies['title_clean'] = movies['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
        movies['genres_list'] = movies['genres'].str.split('|')
        movies['num_genres'] = movies['genres_list'].apply(len)

        # Compute stats
        movie_stats = compute_movie_stats(ratings)
        movies = movies.merge(movie_stats, on='movieId', how='left')

        # Save with _20m suffix
        print(f"Saving to {output_dir}...")
        movies.to_parquet(f"{output_dir}/movies_20m.parquet", index=False)
        ratings.to_parquet(f"{output_dir}/ratings_20m.parquet", index=False)

        print("✓ ML-20M data processing complete!")
        data = {'movies': movies, 'ratings': ratings, 'users': None}
    else:
        data = prepare_ml1m_dataset()

    # Print summary stats
    print("\n=== Dataset Summary ===")
    print(f"Movies: {len(data['movies'])}")
    print(f"Ratings: {len(data['ratings'])}")
    if data['users'] is not None:
        print(f"Users: {len(data['users'])}")
        print(f"Sparsity: {len(data['ratings']) / (len(data['users']) * len(data['movies'])) * 100:.2f}%")
    print(f"\nYear range: {data['movies']['year'].min():.0f} - {data['movies']['year'].max():.0f}")
    print(f"Avg ratings per movie: {data['movies']['num_ratings'].mean():.1f}")
