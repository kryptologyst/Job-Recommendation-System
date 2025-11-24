#!/usr/bin/env python3
"""
Project 353: Modern Job Recommendation System

A comprehensive job recommendation system that combines content-based and 
collaborative filtering approaches to provide personalized job recommendations.

This is a modernized version of the original simple TF-IDF implementation,
now featuring multiple models, proper evaluation, and production-ready structure.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from job_recommender.data.generator import load_data, set_seed
from job_recommender.models.content_based import TFIDFRecommender, SentenceTransformerRecommender
from job_recommender.models.collaborative_filtering import (
    UserBasedCFRecommender, ItemBasedCFRecommender, MatrixFactorizationRecommender
)
from job_recommender.evaluation import RecommendationEvaluator


def demonstrate_original_approach():
    """Demonstrate the original simple TF-IDF approach."""
    print("=" * 80)
    print("ORIGINAL SIMPLE TF-IDF APPROACH")
    print("=" * 80)
    
    # Original simple implementation
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Simple job data (as in original)
    jobs = ['Software Engineer', 'Data Scientist', 'Product Manager', 'UX Designer', 'Marketing Specialist']
    job_descriptions = [
        "Develop software applications, write clean code, and work with other engineers.",
        "Analyze large datasets, build machine learning models, and conduct data analysis.",
        "Lead product development, define product strategy, and work with cross-functional teams.",
        "Design user experiences for web and mobile applications, conducting user research and testing.",
        "Create marketing campaigns, analyze market trends, and optimize digital marketing strategies."
    ]
    
    # User profile
    user_profile = "I have experience in software development, Python programming, and working with machine learning models."
    
    # TF-IDF approach
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(job_descriptions + [user_profile])
    
    # Calculate similarities
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    similar_indices = cosine_sim.argsort()[-3:][::-1]
    
    print("Original TF-IDF Recommendations:")
    for i, idx in enumerate(similar_indices, 1):
        print(f"{i}. {jobs[idx]} (Similarity: {cosine_sim[idx]:.3f})")
    
    print("\nThis simple approach works but has limitations:")
    print("- No user interaction history")
    print("- No evaluation metrics")
    print("- No scalability considerations")
    print("- Limited to content similarity only")


def demonstrate_modern_system():
    """Demonstrate the modern, comprehensive system."""
    print("\n" + "=" * 80)
    print("MODERN COMPREHENSIVE SYSTEM")
    print("=" * 80)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load realistic data
    print("Loading realistic job recommendation data...")
    jobs_df, users_df, interactions_df = load_data()
    
    print(f"Loaded {len(jobs_df)} jobs, {len(users_df)} users, {len(interactions_df)} interactions")
    
    # Train multiple models
    print("\nTraining multiple recommendation models...")
    
    models = {}
    
    # Content-based models
    print("  - Training TF-IDF model...")
    tfidf_model = TFIDFRecommender()
    tfidf_model.fit(interactions_df, jobs_df, users_df)
    models['TF-IDF'] = tfidf_model
    
    print("  - Training User-Based Collaborative Filtering...")
    user_cf_model = UserBasedCFRecommender(min_interactions=3)
    user_cf_model.fit(interactions_df, jobs_df, users_df)
    models['User-Based CF'] = user_cf_model
    
    print("  - Training Item-Based Collaborative Filtering...")
    item_cf_model = ItemBasedCFRecommender(min_interactions=3)
    item_cf_model.fit(interactions_df, jobs_df, users_df)
    models['Item-Based CF'] = item_cf_model
    
    print("  - Training Matrix Factorization...")
    mf_model = MatrixFactorizationRecommender(n_factors=30, min_interactions=3)
    mf_model.fit(interactions_df, jobs_df, users_df)
    models['Matrix Factorization'] = mf_model
    
    # Demonstrate recommendations
    print("\nGenerating recommendations for a sample user...")
    sample_user = users_df['user_id'].iloc[0]
    user_info = users_df[users_df['user_id'] == sample_user].iloc[0]
    
    print(f"\nUser Profile:")
    print(f"  Name: {user_info['name']}")
    print(f"  Skills: {user_info['skills']}")
    print(f"  Experience: {user_info['experience_years']} years")
    print(f"  Location: {user_info['location']}")
    print(f"  Salary Expectation: ${user_info['salary_expectation']:,}")
    
    # Show user's interaction history
    user_interactions = interactions_df[interactions_df['user_id'] == sample_user]
    if len(user_interactions) > 0:
        print(f"\nUser's Interaction History:")
        for _, interaction in user_interactions.head(3).iterrows():
            job_info = jobs_df[jobs_df['job_id'] == interaction['job_id']].iloc[0]
            print(f"  - {job_info['title']} at {job_info['company']} ({interaction['interaction_type']})")
    
    # Generate recommendations from different models
    print(f"\nRecommendations from different models:")
    print("-" * 60)
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        try:
            recommendations = model.recommend(sample_user, n_recommendations=3)
            for i, rec in enumerate(recommendations, 1):
                job_info = jobs_df[jobs_df['job_id'] == rec['job_id']].iloc[0]
                print(f"  {i}. {rec['title']} at {rec['company']} (Score: {rec['score']:.3f})")
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    # Demonstrate job similarity
    print(f"\nJob Similarity Analysis:")
    print("-" * 40)
    
    sample_job = jobs_df['job_id'].iloc[0]
    job_info = jobs_df[jobs_df['job_id'] == sample_job].iloc[0]
    
    print(f"Finding jobs similar to: {job_info['title']} at {job_info['company']}")
    print(f"Description: {job_info['description']}")
    
    similar_jobs = tfidf_model.get_similar_jobs(sample_job, n_similar=3)
    print(f"\nSimilar Jobs:")
    for i, similar in enumerate(similar_jobs, 1):
        similar_job_info = jobs_df[jobs_df['job_id'] == similar['job_id']].iloc[0]
        print(f"  {i}. {similar['title']} at {similar['company']} (Similarity: {similar['similarity']:.3f})")
    
    # Show system capabilities
    print(f"\n" + "=" * 80)
    print("SYSTEM CAPABILITIES")
    print("=" * 80)
    
    print("✅ Multiple Recommendation Approaches:")
    print("   - Content-based (TF-IDF, Sentence Transformers)")
    print("   - Collaborative Filtering (User-based, Item-based)")
    print("   - Matrix Factorization")
    
    print("\n✅ Comprehensive Evaluation:")
    print("   - Precision@K, Recall@K, MAP@K, NDCG@K")
    print("   - Hit Rate, Coverage, Diversity metrics")
    print("   - Model comparison and benchmarking")
    
    print("\n✅ Production-Ready Features:")
    print("   - Type hints and documentation")
    print("   - Proper error handling")
    print("   - Configurable parameters")
    print("   - Unit tests and CI/CD")
    print("   - Interactive demo interface")
    
    print("\n✅ Scalability Considerations:")
    print("   - Efficient data structures")
    print("   - FAISS indexing for similarity search")
    print("   - Caching and optimization")
    print("   - Modular architecture")


def main():
    """Main demonstration function."""
    print("Job Recommendation System - Modernization Demo")
    print("=" * 80)
    print("This demo shows the evolution from a simple TF-IDF implementation")
    print("to a comprehensive, production-ready recommendation system.")
    
    # Demonstrate original approach
    demonstrate_original_approach()
    
    # Demonstrate modern system
    demonstrate_modern_system()
    
    print(f"\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Run the interactive demo: streamlit run demo.py")
    print("2. Train and evaluate models: python scripts/train.py")
    print("3. Explore the Jupyter notebook: notebooks/example_usage.ipynb")
    print("4. Run tests: pytest tests/")
    print("5. Check the README.md for detailed documentation")
    
    print(f"\nThe system is now ready for production use with:")
    print("- Clean, maintainable code")
    print("- Comprehensive evaluation")
    print("- Multiple recommendation approaches")
    print("- Interactive user interface")
    print("- Proper testing and CI/CD")


if __name__ == "__main__":
    main()
