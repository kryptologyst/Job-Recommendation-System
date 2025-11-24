"""Evaluation metrics and utilities for recommendation systems."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from collections import defaultdict


class RecommendationEvaluator:
    """Evaluator for recommendation models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.metrics_history = []
    
    def precision_at_k(self, recommendations: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Precision@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        if len(top_k_recs) == 0:
            return 0.0
        
        relevant_recommendations = sum(1 for item in top_k_recs if item in relevant_set)
        return relevant_recommendations / len(top_k_recs)
    
    def recall_at_k(self, recommendations: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Recall@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        relevant_recommendations = sum(1 for item in top_k_recs if item in relevant_set)
        return relevant_recommendations / len(relevant_items)
    
    def map_at_k(self, recommendations: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Mean Average Precision@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        if len(top_k_recs) == 0:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recs):
            if item in relevant_set:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant_items) if len(relevant_items) > 0 else 0.0
    
    def ndcg_at_k(self, recommendations: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recs):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def hit_rate_at_k(self, recommendations: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Hit Rate@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        return 1.0 if any(item in relevant_set for item in top_k_recs) else 0.0
    
    def coverage(self, recommendations: List[List[str]], catalog_size: int) -> float:
        """Calculate catalog coverage.
        
        Args:
            recommendations: List of recommendation lists for each user
            catalog_size: Total number of items in catalog
            
        Returns:
            Coverage score
        """
        if catalog_size == 0:
            return 0.0
        
        all_recommended = set()
        for recs in recommendations:
            all_recommended.update(recs)
        
        return len(all_recommended) / catalog_size
    
    def diversity(self, recommendations: List[List[str]], similarity_matrix: Optional[np.ndarray] = None) -> float:
        """Calculate intra-list diversity.
        
        Args:
            recommendations: List of recommendation lists for each user
            similarity_matrix: Optional similarity matrix between items
            
        Returns:
            Diversity score
        """
        if similarity_matrix is None:
            # Simple diversity based on unique items
            total_items = sum(len(recs) for recs in recommendations)
            unique_items = len(set(item for recs in recommendations for item in recs))
            return unique_items / total_items if total_items > 0 else 0.0
        
        diversity_scores = []
        for recs in recommendations:
            if len(recs) <= 1:
                diversity_scores.append(1.0)
                continue
            
            # Calculate average pairwise dissimilarity
            total_similarity = 0.0
            pairs = 0
            
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    # Assuming item IDs can be mapped to matrix indices
                    # This is a simplified version - in practice, you'd need proper mapping
                    total_similarity += similarity_matrix[i, j] if i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1] else 0.0
                    pairs += 1
            
            avg_similarity = total_similarity / pairs if pairs > 0 else 0.0
            diversity_scores.append(1.0 - avg_similarity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def evaluate_model(self, model, test_data: pd.DataFrame, jobs_df: pd.DataFrame, 
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """Evaluate a recommendation model.
        
        Args:
            model: Trained recommendation model
            test_data: Test interactions DataFrame
            jobs_df: Jobs DataFrame
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        # Group test data by user
        user_test_data = test_data.groupby('user_id')['job_id'].apply(list).to_dict()
        
        # Calculate metrics for each K
        for k in k_values:
            precision_scores = []
            recall_scores = []
            map_scores = []
            ndcg_scores = []
            hit_rate_scores = []
            
            for user_id, relevant_items in user_test_data.items():
                try:
                    # Get recommendations
                    recommendations = model.recommend(user_id, n_recommendations=k, exclude_seen=True)
                    rec_ids = [rec['job_id'] for rec in recommendations]
                    
                    # Calculate metrics
                    precision_scores.append(self.precision_at_k(rec_ids, relevant_items, k))
                    recall_scores.append(self.recall_at_k(rec_ids, relevant_items, k))
                    map_scores.append(self.map_at_k(rec_ids, relevant_items, k))
                    ndcg_scores.append(self.ndcg_at_k(rec_ids, relevant_items, k))
                    hit_rate_scores.append(self.hit_rate_at_k(rec_ids, relevant_items, k))
                    
                except Exception as e:
                    # Handle cold start users or other errors
                    precision_scores.append(0.0)
                    recall_scores.append(0.0)
                    map_scores.append(0.0)
                    ndcg_scores.append(0.0)
                    hit_rate_scores.append(0.0)
            
            results[f'precision@{k}'] = np.mean(precision_scores)
            results[f'recall@{k}'] = np.mean(recall_scores)
            results[f'map@{k}'] = np.mean(map_scores)
            results[f'ndcg@{k}'] = np.mean(ndcg_scores)
            results[f'hit_rate@{k}'] = np.mean(hit_rate_scores)
        
        # Calculate coverage
        all_recommendations = []
        for user_id in user_test_data.keys():
            try:
                recommendations = model.recommend(user_id, n_recommendations=20, exclude_seen=True)
                rec_ids = [rec['job_id'] for rec in recommendations]
                all_recommendations.append(rec_ids)
            except:
                all_recommendations.append([])
        
        results['coverage'] = self.coverage(all_recommendations, len(jobs_df))
        
        return results
    
    def compare_models(self, models: Dict[str, Any], test_data: pd.DataFrame, 
                      jobs_df: pd.DataFrame, k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Compare multiple recommendation models.
        
        Args:
            models: Dictionary of model_name -> model
            test_data: Test interactions DataFrame
            jobs_df: Jobs DataFrame
            k_values: List of K values to evaluate
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            model_results = self.evaluate_model(model, test_data, jobs_df, k_values)
            model_results['model'] = model_name
            results.append(model_results)
        
        return pd.DataFrame(results)
