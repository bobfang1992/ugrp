"""
Base recommender model using ALS (Alternating Least Squares).
Uses the implicit library for fast matrix factorization.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import pickle
from pathlib import Path


class ALSRecommender:
    """
    Matrix Factorization recommender using ALS.

    Following spec §5: Fixed baseline recommender for candidate generation.
    """

    def __init__(self, factors=64, regularization=0.01, iterations=15, random_state=42):
        """
        Args:
            factors: Number of latent factors
            regularization: L2 regularization strength
            iterations: Number of ALS iterations
            random_state: Random seed for reproducibility
        """
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state
        )
        self.user_id_map = {}  # original userId -> internal index
        self.movie_id_map = {}  # original movieId -> internal index
        self.reverse_user_map = {}  # internal index -> original userId
        self.reverse_movie_map = {}  # internal index -> original movieId
        self.user_item_matrix = None

    def _create_mappings(self, ratings_df):
        """Create user and movie ID mappings"""
        unique_users = sorted(ratings_df['userId'].unique())
        unique_movies = sorted(ratings_df['movieId'].unique())

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_movie_map = {idx: mid for mid, idx in self.movie_id_map.items()}

    def _build_user_item_matrix(self, ratings_df):
        """Build sparse user-item interaction matrix"""
        # Map to internal indices
        user_indices = ratings_df['userId'].map(self.user_id_map)
        movie_indices = ratings_df['movieId'].map(self.movie_id_map)

        # Use ratings as values (implicit library expects this format)
        # Shape: (n_users, n_items)
        matrix = csr_matrix(
            (ratings_df['rating'].values, (user_indices, movie_indices)),
            shape=(len(self.user_id_map), len(self.movie_id_map))
        )
        return matrix

    def fit(self, ratings_df):
        """
        Train the ALS model on ratings data.

        Args:
            ratings_df: DataFrame with columns [userId, movieId, rating]
        """
        print("Creating ID mappings...")
        self._create_mappings(ratings_df)

        print("Building user-item matrix...")
        self.user_item_matrix = self._build_user_item_matrix(ratings_df)

        print(f"Matrix shape: {self.user_item_matrix.shape}")
        print(f"Sparsity: {self.user_item_matrix.nnz / np.prod(self.user_item_matrix.shape) * 100:.2f}%")

        print("Training ALS model...")
        self.model.fit(self.user_item_matrix)

        print("✓ Training complete!")

    def get_candidates(self, user_id, n=200, filter_already_liked=True):
        """
        Get top-N candidate items for a user.

        Args:
            user_id: Original user ID
            n: Number of candidates to return
            filter_already_liked: Whether to exclude items user has already rated

        Returns:
            List of (movieId, score) tuples
        """
        if user_id not in self.user_id_map:
            raise ValueError(f"User {user_id} not found in training data")

        user_idx = self.user_id_map[user_id]

        # Get recommendations from model
        # recommend() returns (item_ids, scores)
        item_ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.user_item_matrix[user_idx],
            N=n,
            filter_already_liked_items=filter_already_liked
        )

        # Map back to original movie IDs
        candidates = [
            (self.reverse_movie_map[item_idx], float(score))
            for item_idx, score in zip(item_ids, scores)
        ]

        return candidates

    def get_all_candidates(self, n=200, filter_already_liked=True):
        """
        Generate top-N candidates for all users.

        Returns:
            DataFrame with columns [userId, movieId, score, rank]
        """
        print(f"Generating top-{n} candidates for {len(self.user_id_map)} users...")

        all_candidates = []

        for user_id in self.user_id_map.keys():
            candidates = self.get_candidates(user_id, n, filter_already_liked)
            for rank, (movie_id, score) in enumerate(candidates, 1):
                all_candidates.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'score': score,
                    'rank': rank
                })

        candidates_df = pd.DataFrame(all_candidates)
        print(f"✓ Generated {len(candidates_df)} candidate recommendations")

        return candidates_df

    def save(self, path):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'user_id_map': self.user_id_map,
                'movie_id_map': self.movie_id_map,
                'reverse_user_map': self.reverse_user_map,
                'reverse_movie_map': self.reverse_movie_map,
                'user_item_matrix': self.user_item_matrix
            }, f)
        print(f"✓ Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        recommender = cls()
        recommender.model = data['model']
        recommender.user_id_map = data['user_id_map']
        recommender.movie_id_map = data['movie_id_map']
        recommender.reverse_user_map = data['reverse_user_map']
        recommender.reverse_movie_map = data['reverse_movie_map']
        recommender.user_item_matrix = data['user_item_matrix']

        print(f"✓ Model loaded from {path}")
        return recommender


if __name__ == "__main__":
    # Load processed data
    print("Loading ratings data...")
    ratings = pd.read_parquet("data/processed/ratings.parquet")

    # Train model
    recommender = ALSRecommender(factors=64, iterations=15)
    recommender.fit(ratings)

    # Generate candidates
    candidates_df = recommender.get_all_candidates(n=200)

    # Save
    recommender.save("data/processed/als_model.pkl")
    candidates_df.to_parquet("data/processed/candidates.parquet", index=False)

    # Quick test
    print("\n=== Sample recommendations for user 1 ===")
    test_candidates = recommender.get_candidates(user_id=1, n=10)
    movies = pd.read_parquet("data/processed/movies.parquet")
    for movie_id, score in test_candidates:
        movie = movies[movies['movieId'] == movie_id].iloc[0]
        print(f"{score:.3f} - {movie['title_clean']} ({movie['year']:.0f}) [{movie['genres']}]")
