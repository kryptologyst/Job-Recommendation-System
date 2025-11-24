# Job Recommendation System

A production-ready job recommendation system that combines content-based and collaborative filtering approaches to provide personalized job recommendations.

## Features

- **Multiple Recommendation Approaches**: Content-based (TF-IDF, Sentence Transformers) and Collaborative Filtering (User-based, Item-based, Matrix Factorization)
- **Comprehensive Evaluation**: Precision@K, Recall@K, MAP@K, NDCG@K, Hit Rate, Coverage, and Diversity metrics
- **Interactive Demo**: Streamlit-based web interface for exploring recommendations
- **Realistic Data Generation**: Synthetic but realistic job and user data with proper relationships
- **Production-Ready Structure**: Clean code with type hints, proper documentation, and testing framework

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Job-Recommendation-System.git
cd Job-Recommendation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Demo

1. Start the Streamlit demo:
```bash
streamlit run demo.py
```

2. Open your browser to `http://localhost:8501` and explore the interactive interface.

### Training Models

1. Train all models and evaluate performance:
```bash
python scripts/train.py --config configs/config.yaml
```

2. View results in `models/evaluation_results.csv`

## Project Structure

```
job-recommendation-system/
├── src/
│   └── job_recommender/
│       ├── data/
│       │   ├── __init__.py
│       │   └── generator.py          # Data generation and loading
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py               # Base recommender interface
│       │   ├── content_based.py      # TF-IDF and SentenceTransformer models
│       │   └── collaborative_filtering.py  # CF models
│       ├── evaluation/
│       │   └── __init__.py           # Evaluation metrics
│       └── utils/
├── configs/
│   └── config.yaml                   # Configuration file
├── scripts/
│   └── train.py                      # Training script
├── tests/                            # Unit tests
├── notebooks/                        # Jupyter notebooks for analysis
├── data/                             # Generated data files
├── models/                           # Trained model outputs
├── demo.py                           # Streamlit demo
├── requirements.txt                  # Python dependencies
├── pyproject.toml                   # Project configuration
└── README.md
```

## Data Schema

### Jobs Dataset (`jobs.csv`)
- `job_id`: Unique job identifier
- `title`: Job title
- `description`: Job description
- `required_skills`: Comma-separated list of required skills
- `experience_level`: Entry, Mid, Senior, Lead
- `location`: Job location
- `salary_min`, `salary_max`: Salary range
- `company`: Company name
- `posted_date`: When the job was posted
- `job_type`: Full-time, Part-time, Contract, Internship
- `remote_allowed`: Boolean for remote work

### Users Dataset (`users.csv`)
- `user_id`: Unique user identifier
- `name`: User name
- `skills`: Comma-separated list of user skills
- `experience_years`: Years of experience
- `education_level`: Education level
- `location`: Current location
- `preferred_location`: Preferred work location
- `salary_expectation`: Expected salary
- `job_type_preference`: Preferred job type
- `remote_preference`: Remote work preference

### Interactions Dataset (`interactions.csv`)
- `user_id`: User identifier
- `job_id`: Job identifier
- `interaction_type`: apply, view, save
- `weight`: Interaction weight (0-1)
- `timestamp`: When the interaction occurred
- `compatibility_score`: Calculated compatibility score

## Models

### Content-Based Models

1. **TF-IDF Recommender**: Uses TF-IDF vectorization to create job and user profiles, then recommends based on cosine similarity.

2. **SentenceTransformer Recommender**: Uses pre-trained sentence transformers to create dense embeddings for better semantic understanding.

### Collaborative Filtering Models

1. **User-Based CF**: Finds similar users and recommends jobs that similar users have interacted with.

2. **Item-Based CF**: Finds similar jobs and recommends jobs similar to ones the user has interacted with.

3. **Matrix Factorization**: Uses SVD to decompose the user-item interaction matrix into latent factors.

## Evaluation Metrics

- **Precision@K**: Proportion of relevant items in top-K recommendations
- **Recall@K**: Proportion of relevant items that were recommended
- **MAP@K**: Mean Average Precision across all users
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate@K**: Proportion of users with at least one relevant recommendation
- **Coverage**: Proportion of catalog items that can be recommended
- **Diversity**: Intra-list diversity of recommendations

## Configuration

The system is configured via `configs/config.yaml`:

```yaml
# Data settings
data:
  test_size: 0.2
  min_interactions_per_user: 3
  min_interactions_per_job: 2

# Model configurations
models:
  tfidf:
    enabled: true
    max_features: 1000
    ngram_range: [1, 2]
  
  sentence_transformer:
    enabled: true
    model_name: "all-MiniLM-L6-v2"
  
  user_based_cf:
    enabled: true
    min_interactions: 5
  
  item_based_cf:
    enabled: true
    min_interactions: 5
  
  matrix_factorization:
    enabled: true
    n_factors: 50
    min_interactions: 5

# Evaluation settings
evaluation:
  k_values: [5, 10, 20]
  metrics: ["precision", "recall", "map", "ndcg", "hit_rate", "coverage"]
```

## Usage Examples

### Basic Training and Evaluation

```python
from src.job_recommender.data.generator import load_data
from src.job_recommender.models.content_based import TFIDFRecommender
from src.job_recommender.evaluation import RecommendationEvaluator

# Load data
jobs_df, users_df, interactions_df = load_data()

# Train model
model = TFIDFRecommender()
model.fit(interactions_df, jobs_df, users_df)

# Generate recommendations
recommendations = model.recommend("user_0001", n_recommendations=10)
print(recommendations)
```

### Model Comparison

```python
from src.job_recommender.models.content_based import TFIDFRecommender, SentenceTransformerRecommender
from src.job_recommender.models.collaborative_filtering import UserBasedCFRecommender

# Train multiple models
models = {
    "TF-IDF": TFIDFRecommender(),
    "SentenceTransformer": SentenceTransformerRecommender(),
    "User-Based CF": UserBasedCFRecommender()
}

for name, model in models.items():
    model.fit(interactions_df, jobs_df, users_df)

# Evaluate and compare
evaluator = RecommendationEvaluator()
results = evaluator.compare_models(models, test_df, jobs_df)
print(results)
```

## Development

### Code Quality

The project uses modern Python development practices:

- **Type hints**: All functions include proper type annotations
- **Documentation**: Google-style docstrings for all classes and functions
- **Code formatting**: Black for code formatting, Ruff for linting
- **Testing**: pytest for unit tests

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ demo.py
ruff check src/ scripts/ demo.py
```

## Performance Considerations

- **Scalability**: The current implementation is suitable for datasets with up to 100K users and 10K jobs
- **Memory Usage**: SentenceTransformer models require significant memory for large datasets
- **Cold Start**: The system handles cold start users by recommending popular items
- **Real-time Recommendations**: For production use, consider implementing caching and model serving

## Future Enhancements

- **Hybrid Models**: Combine content-based and collaborative filtering approaches
- **Deep Learning**: Implement neural collaborative filtering or deep learning models
- **Real-time Updates**: Support for real-time model updates with new interactions
- **A/B Testing**: Framework for testing different recommendation strategies
- **Fairness Metrics**: Evaluation of recommendation fairness across user groups
- **Multi-objective Optimization**: Balance accuracy, diversity, and novelty

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using scikit-learn, sentence-transformers, and other open-source libraries
- Inspired by modern recommendation system research and best practices
- Data generation approach based on realistic job market patterns
# Job-Recommendation-System
