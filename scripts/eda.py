"""Quick EDA on processed ML-1M data"""

import pandas as pd
from pathlib import Path

# Load processed data
data_dir = Path("data/processed")
movies = pd.read_parquet(data_dir / "movies.parquet")
ratings = pd.read_parquet(data_dir / "ratings.parquet")
users = pd.read_parquet(data_dir / "users.parquet")

print("=== ML-1M Dataset Overview ===\n")

# Basic stats
print(f"Movies: {len(movies):,}")
print(f"Ratings: {len(ratings):,}")
print(f"Users: {len(users):,}")
print(f"Sparsity: {len(ratings) / (len(users) * len(movies)) * 100:.2f}%\n")

# Ratings distribution
print("=== Rating Distribution ===")
print(ratings['rating'].value_counts().sort_index())
print(f"Mean rating: {ratings['rating'].mean():.2f}")
print(f"Std rating: {ratings['rating'].std():.2f}\n")

# Year distribution
print("=== Year Distribution ===")
print(f"Range: {movies['year'].min():.0f} - {movies['year'].max():.0f}")
print(f"Median: {movies['year'].median():.0f}")
print("\nMovies per decade:")
movies['decade'] = (movies['year'] // 10) * 10
print(movies['decade'].value_counts().sort_index()[:10])
print()

# Genre distribution
print("=== Genre Distribution ===")
from collections import Counter
all_genres = []
for genres_list in movies['genres_list']:
    all_genres.extend(genres_list)
genre_counts = Counter(all_genres)
print(f"Unique genres: {len(genre_counts)}")
print("\nTop 10 genres:")
for genre, count in genre_counts.most_common(10):
    print(f"  {genre:20s} {count:4d} ({count/len(movies)*100:.1f}%)")
print()

# Popularity distribution
print("=== Popularity Distribution ===")
print(f"Min ratings per movie: {movies['num_ratings'].min():.0f}")
print(f"Max ratings per movie: {movies['num_ratings'].max():.0f}")
print(f"Median ratings per movie: {movies['num_ratings'].median():.0f}")
print(f"Mean ratings per movie: {movies['num_ratings'].mean():.1f}")
print("\nMost popular movies:")
print(movies.nlargest(5, 'num_ratings')[['title_clean', 'year', 'num_ratings', 'avg_rating']])
print()

# User activity
print("=== User Activity ===")
user_activity = ratings.groupby('userId').size()
print(f"Min ratings per user: {user_activity.min()}")
print(f"Max ratings per user: {user_activity.max()}")
print(f"Median ratings per user: {user_activity.median():.0f}")
print(f"Mean ratings per user: {user_activity.mean():.1f}")
