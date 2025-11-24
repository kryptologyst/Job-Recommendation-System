"""Unit tests for job recommendation system."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.job_recommender.data.generator import generate_job_data, generate_user_data, generate_interactions
from src.job_recommender.models.content_based import TFIDFRecommender
from src.job_recommender.models.collaborative_filtering import UserBasedCFRecommender
from src.job_recommender.evaluation import RecommendationEvaluator


class TestDataGenerator:
    """Test data generation functions."""
    
    def test_generate_job_data(self):
        """Test job data generation."""
        jobs_df = generate_job_data(n_jobs=10)
        
        assert len(jobs_df) == 10
        assert 'job_id' in jobs_df.columns
        assert 'title' in jobs_df.columns
        assert 'description' in jobs_df.columns
        assert 'required_skills' in jobs_df.columns
        assert all(jobs_df['job_id'].str.startswith('job_'))
    
    def test_generate_user_data(self):
        """Test user data generation."""
        users_df = generate_user_data(n_users=5)
        
        assert len(users_df) == 5
        assert 'user_id' in users_df.columns
        assert 'name' in users_df.columns
        assert 'skills' in users_df.columns
        assert all(users_df['user_id'].str.startswith('user_'))
    
    def test_generate_interactions(self):
        """Test interaction data generation."""
        jobs_df = generate_job_data(n_jobs=5)
        users_df = generate_user_data(n_users=3)
        interactions_df = generate_interactions(jobs_df, users_df, n_interactions=10)
        
        assert len(interactions_df) == 10
        assert 'user_id' in interactions_df.columns
        assert 'job_id' in interactions_df.columns
        assert 'interaction_type' in interactions_df.columns
        assert 'weight' in interactions_df.columns
        assert all(interactions_df['weight'] >= 0)
        assert all(interactions_df['weight'] <= 1)


class TestTFIDFRecommender:
    """Test TF-IDF recommender."""
    
    def setup_method(self):
        """Set up test data."""
        self.jobs_df = generate_job_data(n_jobs=20)
        self.users_df = generate_user_data(n_users=10)
        self.interactions_df = generate_interactions(self.jobs_df, self.users_df, n_interactions=50)
        self.model = TFIDFRecommender()
    
    def test_fit(self):
        """Test model fitting."""
        self.model.fit(self.interactions_df, self.jobs_df, self.users_df)
        assert self.model.is_fitted
        assert self.model.vectorizer is not None
        assert self.model.job_vectors is not None
    
    def test_recommend(self):
        """Test recommendation generation."""
        self.model.fit(self.interactions_df, self.jobs_df, self.users_df)
        
        # Test with existing user
        user_id = self.interactions_df['user_id'].iloc[0]
        recommendations = self.model.recommend(user_id, n_recommendations=5)
        
        assert len(recommendations) <= 5
        assert all('job_id' in rec for rec in recommendations)
        assert all('score' in rec for rec in recommendations)
        assert all('title' in rec for rec in recommendations)
    
    def test_cold_start(self):
        """Test cold start user handling."""
        self.model.fit(self.interactions_df, self.jobs_df, self.users_df)
        
        # Test with non-existent user
        recommendations = self.model.recommend("non_existent_user", n_recommendations=5)
        
        assert len(recommendations) <= 5
        assert all('job_id' in rec for rec in recommendations)
    
    def test_get_similar_jobs(self):
        """Test similar job finding."""
        self.model.fit(self.interactions_df, self.jobs_df, self.users_df)
        
        job_id = self.jobs_df['job_id'].iloc[0]
        similar_jobs = self.model.get_similar_jobs(job_id, n_similar=5)
        
        assert len(similar_jobs) <= 5
        assert all('job_id' in job for job in similar_jobs)
        assert all('similarity' in job for job in similar_jobs)


class TestUserBasedCFRecommender:
    """Test User-Based CF recommender."""
    
    def setup_method(self):
        """Set up test data."""
        self.jobs_df = generate_job_data(n_jobs=20)
        self.users_df = generate_user_data(n_users=10)
        self.interactions_df = generate_interactions(self.jobs_df, self.users_df, n_interactions=100)
        self.model = UserBasedCFRecommender(min_interactions=2)
    
    def test_fit(self):
        """Test model fitting."""
        self.model.fit(self.interactions_df, self.jobs_df, self.users_df)
        assert self.model.is_fitted
        assert self.model.user_item_matrix is not None
        assert self.model.user_similarity_matrix is not None
    
    def test_recommend(self):
        """Test recommendation generation."""
        self.model.fit(self.interactions_df, self.jobs_df, self.users_df)
        
        # Test with existing user
        user_id = self.interactions_df['user_id'].iloc[0]
        recommendations = self.model.recommend(user_id, n_recommendations=5)
        
        assert len(recommendations) <= 5
        assert all('job_id' in rec for rec in recommendations)
        assert all('score' in rec for rec in recommendations)


class TestRecommendationEvaluator:
    """Test recommendation evaluator."""
    
    def setup_method(self):
        """Set up test data."""
        self.evaluator = RecommendationEvaluator()
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        precision = self.evaluator.precision_at_k(recommendations, relevant_items, k=5)
        assert precision == 2/5  # 2 relevant items out of 5 recommendations
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        recall = self.evaluator.recall_at_k(recommendations, relevant_items, k=5)
        assert recall == 2/3  # 2 relevant items found out of 3 total relevant items
    
    def test_map_at_k(self):
        """Test MAP@k calculation."""
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        map_score = self.evaluator.map_at_k(recommendations, relevant_items, k=5)
        assert map_score > 0
        assert map_score <= 1
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        ndcg = self.evaluator.ndcg_at_k(recommendations, relevant_items, k=5)
        assert ndcg >= 0
        assert ndcg <= 1
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        hit_rate = self.evaluator.hit_rate_at_k(recommendations, relevant_items, k=5)
        assert hit_rate == 1.0  # At least one relevant item found
    
    def test_coverage(self):
        """Test coverage calculation."""
        recommendations = [
            ['item1', 'item2', 'item3'],
            ['item2', 'item3', 'item4'],
            ['item3', 'item4', 'item5']
        ]
        catalog_size = 10
        
        coverage = self.evaluator.coverage(recommendations, catalog_size)
        assert coverage == 5/10  # 5 unique items out of 10 total


if __name__ == "__main__":
    pytest.main([__file__])
