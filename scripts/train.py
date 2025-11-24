"""Main training script for job recommendation system."""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.job_recommender.data.generator import load_data, set_seed
from src.job_recommender.models.content_based import TFIDFRecommender, SentenceTransformerRecommender
from src.job_recommender.models.collaborative_filtering import (
    UserBasedCFRecommender, ItemBasedCFRecommender, MatrixFactorizationRecommender
)
from src.job_recommender.evaluation import RecommendationEvaluator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def split_data(interactions_df: pd.DataFrame, test_size: float = 0.2, 
               random_state: int = 42) -> tuple:
    """Split interactions data into train and test sets.
    
    Args:
        interactions_df: Interactions DataFrame
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Sort by timestamp to ensure temporal split
    interactions_df = interactions_df.sort_values('timestamp')
    
    # Split by user to ensure each user appears in only one set
    users = interactions_df['user_id'].unique()
    train_users, test_users = train_test_split(
        users, test_size=test_size, random_state=random_state
    )
    
    train_df = interactions_df[interactions_df['user_id'].isin(train_users)]
    test_df = interactions_df[interactions_df['user_id'].isin(test_users)]
    
    return train_df, test_df


def train_models(config: Dict[str, Any], train_df: pd.DataFrame, 
                jobs_df: pd.DataFrame, users_df: pd.DataFrame) -> Dict[str, Any]:
    """Train all recommendation models.
    
    Args:
        config: Configuration dictionary
        train_df: Training interactions DataFrame
        jobs_df: Jobs DataFrame
        users_df: Users DataFrame
        
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    # Content-based models
    if config.get('models', {}).get('tfidf', {}).get('enabled', True):
        print("Training TF-IDF model...")
        tfidf_model = TFIDFRecommender(
            max_features=config['models']['tfidf'].get('max_features', 1000),
            ngram_range=tuple(config['models']['tfidf'].get('ngram_range', [1, 2]))
        )
        tfidf_model.fit(train_df, jobs_df, users_df)
        models['TF-IDF'] = tfidf_model
    
    if config.get('models', {}).get('sentence_transformer', {}).get('enabled', True):
        print("Training SentenceTransformer model...")
        st_model = SentenceTransformerRecommender(
            model_name=config['models']['sentence_transformer'].get('model_name', 'all-MiniLM-L6-v2')
        )
        st_model.fit(train_df, jobs_df, users_df)
        models['SentenceTransformer'] = st_model
    
    # Collaborative filtering models
    if config.get('models', {}).get('user_based_cf', {}).get('enabled', True):
        print("Training User-Based CF model...")
        user_cf_model = UserBasedCFRecommender(
            min_interactions=config['models']['user_based_cf'].get('min_interactions', 5)
        )
        user_cf_model.fit(train_df, jobs_df, users_df)
        models['User-Based CF'] = user_cf_model
    
    if config.get('models', {}).get('item_based_cf', {}).get('enabled', True):
        print("Training Item-Based CF model...")
        item_cf_model = ItemBasedCFRecommender(
            min_interactions=config['models']['item_based_cf'].get('min_interactions', 5)
        )
        item_cf_model.fit(train_df, jobs_df, users_df)
        models['Item-Based CF'] = item_cf_model
    
    if config.get('models', {}).get('matrix_factorization', {}).get('enabled', True):
        print("Training Matrix Factorization model...")
        mf_model = MatrixFactorizationRecommender(
            n_factors=config['models']['matrix_factorization'].get('n_factors', 50),
            min_interactions=config['models']['matrix_factorization'].get('min_interactions', 5)
        )
        mf_model.fit(train_df, jobs_df, users_df)
        models['Matrix Factorization'] = mf_model
    
    return models


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train job recommendation models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing data files')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    jobs_df, users_df, interactions_df = load_data(args.data_dir)
    
    print(f"Loaded {len(jobs_df)} jobs, {len(users_df)} users, {len(interactions_df)} interactions")
    
    # Split data
    print("Splitting data...")
    train_df, test_df = split_data(
        interactions_df, 
        test_size=config.get('data', {}).get('test_size', 0.2),
        random_state=args.seed
    )
    
    print(f"Train set: {len(train_df)} interactions")
    print(f"Test set: {len(test_df)} interactions")
    
    # Train models
    print("Training models...")
    models = train_models(config, train_df, jobs_df, users_df)
    
    # Evaluate models
    print("Evaluating models...")
    evaluator = RecommendationEvaluator()
    results_df = evaluator.compare_models(
        models, test_df, jobs_df, 
        k_values=config.get('evaluation', {}).get('k_values', [5, 10, 20])
    )
    
    # Save results
    results_path = output_dir / 'evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation results saved to {results_path}")
    
    # Print results
    print("\nModel Comparison Results:")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Save models (simplified - in practice, you'd use proper serialization)
    for model_name, model in models.items():
        model_path = output_dir / f'{model_name.lower().replace(" ", "_")}_model.pkl'
        # In practice, you'd use joblib.dump() or similar
        print(f"Model {model_name} would be saved to {model_path}")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
