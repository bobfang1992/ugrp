"""
Evaluation metrics for recommender systems.

Implements ranking metrics: Precision@K, Recall@K, NDCG@K, Hit Rate@K, MAP@K.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path


def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Precision@K: Fraction of recommended items that are relevant.

    Args:
        recommended: List of recommended item IDs (in ranking order)
        relevant: List of relevant (ground truth) item IDs
        k: Number of top recommendations to consider

    Returns:
        Precision@K score
    """
    if k == 0:
        return 0.0

    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    hits = len([item for item in recommended_k if item in relevant_set])
    return hits / k


def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Recall@K: Fraction of relevant items that are recommended.

    Args:
        recommended: List of recommended item IDs
        relevant: List of relevant (ground truth) item IDs
        k: Number of top recommendations to consider

    Returns:
        Recall@K score
    """
    if len(relevant) == 0:
        return 0.0

    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    hits = len([item for item in recommended_k if item in relevant_set])
    return hits / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain.

    Measures ranking quality - items higher in the list get more weight.

    Args:
        recommended: List of recommended item IDs (in ranking order)
        relevant: List of relevant (ground truth) item IDs
        k: Number of top recommendations to consider

    Returns:
        NDCG@K score
    """
    if len(relevant) == 0:
        return 0.0

    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    # DCG: Sum of (relevance / log2(position + 1))
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # +2 because i starts at 0

    # IDCG: DCG of perfect ranking
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Hit Rate@K: Binary metric - 1 if any relevant item in top-K, 0 otherwise.

    Args:
        recommended: List of recommended item IDs
        relevant: List of relevant (ground truth) item IDs
        k: Number of top recommendations to consider

    Returns:
        1.0 if hit, 0.0 otherwise
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    return 1.0 if any(item in relevant_set for item in recommended_k) else 0.0


def average_precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Average Precision@K: Mean of precision values at each relevant item position.

    Args:
        recommended: List of recommended item IDs (in ranking order)
        relevant: List of relevant (ground truth) item IDs
        k: Number of top recommendations to consider

    Returns:
        AP@K score
    """
    if len(relevant) == 0:
        return 0.0

    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    precisions = []
    num_hits = 0

    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            num_hits += 1
            precisions.append(num_hits / (i + 1))

    return np.mean(precisions) if precisions else 0.0


def evaluate_model(model, test_df: pd.DataFrame, k_values: List[int] = [10, 20, 50]) -> Dict:
    """
    Evaluate recommender model on test set.

    Args:
        model: Trained ALSRecommender model
        test_df: Test ratings DataFrame with columns [userId, movieId]
        k_values: List of K values to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating model on {len(test_df['userId'].unique())} test users...")

    # Group test items by user
    test_users = test_df.groupby('userId')['movieId'].apply(list).to_dict()

    # Initialize metric accumulators
    metrics = {k: {
        'precision': [],
        'recall': [],
        'ndcg': [],
        'hit_rate': [],
        'ap': []
    } for k in k_values}

    evaluated_users = 0

    for user_id, relevant_items in test_users.items():
        if user_id not in model.user_id_map:
            continue  # Skip users not in training data

        try:
            # Get recommendations
            recommendations = model.get_candidates(user_id, n=max(k_values), filter_already_liked=True)
            recommended_ids = [movie_id for movie_id, score in recommendations]

            # Calculate metrics for each K
            for k in k_values:
                metrics[k]['precision'].append(precision_at_k(recommended_ids, relevant_items, k))
                metrics[k]['recall'].append(recall_at_k(recommended_ids, relevant_items, k))
                metrics[k]['ndcg'].append(ndcg_at_k(recommended_ids, relevant_items, k))
                metrics[k]['hit_rate'].append(hit_rate_at_k(recommended_ids, relevant_items, k))
                metrics[k]['ap'].append(average_precision_at_k(recommended_ids, relevant_items, k))

            evaluated_users += 1

            if evaluated_users % 1000 == 0:
                print(f"  Evaluated {evaluated_users} users...")

        except Exception as e:
            print(f"Warning: Error evaluating user {user_id}: {e}")
            continue

    # Compute averages
    results = {}
    for k in k_values:
        results[f'P@{k}'] = float(np.mean(metrics[k]['precision']))
        results[f'R@{k}'] = float(np.mean(metrics[k]['recall']))
        results[f'NDCG@{k}'] = float(np.mean(metrics[k]['ndcg']))
        results[f'HR@{k}'] = float(np.mean(metrics[k]['hit_rate']))
        results[f'MAP@{k}'] = float(np.mean(metrics[k]['ap']))

    results['num_evaluated_users'] = evaluated_users

    print(f"✓ Evaluation complete on {evaluated_users} users")

    return results


def print_evaluation_results(results: Dict):
    """Pretty print evaluation results."""
    print("\n=== Evaluation Results ===")
    print(f"Evaluated users: {results['num_evaluated_users']}")
    print()

    # Extract K values
    k_values = sorted(set(int(key.split('@')[1]) for key in results.keys() if '@' in key))

    for k in k_values:
        print(f"K = {k}:")
        print(f"  Precision@{k}:  {results[f'P@{k}']:.4f}")
        print(f"  Recall@{k}:     {results[f'R@{k}']:.4f}")
        print(f"  NDCG@{k}:       {results[f'NDCG@{k}']:.4f}")
        print(f"  Hit Rate@{k}:   {results[f'HR@{k}']:.4f}")
        print(f"  MAP@{k}:        {results[f'MAP@{k}']:.4f}")
        print()


def save_evaluation_results(results: Dict, output_path: str):
    """Save evaluation results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved evaluation results to {output_path}")


def load_evaluation_results(path: str) -> Dict:
    """Load evaluation results from JSON."""
    with open(path, 'r') as f:
        results = json.load(f)

    return results
