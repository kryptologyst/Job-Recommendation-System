"""Data generation and loading utilities for job recommendation system."""

import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def generate_job_data(n_jobs: int = 1000, output_path: Optional[str] = None) -> pd.DataFrame:
    """Generate realistic job data with various attributes.
    
    Args:
        n_jobs: Number of jobs to generate
        output_path: Optional path to save the data
        
    Returns:
        DataFrame with job information
    """
    set_seed(42)
    
    # Job titles and their corresponding descriptions
    job_templates = {
        "Software Engineer": {
            "skills": ["Python", "Java", "JavaScript", "React", "Node.js", "SQL", "Git"],
            "description": "Develop software applications, write clean code, and work with other engineers. Experience with modern frameworks and agile development.",
            "experience_level": ["Entry", "Mid", "Senior"],
            "location": ["San Francisco", "New York", "Seattle", "Austin", "Remote"],
            "salary_range": (80000, 180000)
        },
        "Data Scientist": {
            "skills": ["Python", "R", "SQL", "Machine Learning", "Statistics", "Pandas", "Scikit-learn"],
            "description": "Analyze large datasets, build machine learning models, and conduct data analysis. Strong background in statistics and programming.",
            "experience_level": ["Entry", "Mid", "Senior"],
            "location": ["San Francisco", "New York", "Boston", "Seattle", "Remote"],
            "salary_range": (90000, 200000)
        },
        "Product Manager": {
            "skills": ["Product Strategy", "Agile", "Analytics", "User Research", "Leadership", "Communication"],
            "description": "Lead product development, define product strategy, and work with cross-functional teams. Strong analytical and leadership skills.",
            "experience_level": ["Mid", "Senior", "Lead"],
            "location": ["San Francisco", "New York", "Seattle", "Austin", "Remote"],
            "salary_range": (100000, 220000)
        },
        "UX Designer": {
            "skills": ["Figma", "Sketch", "User Research", "Prototyping", "Design Systems", "Adobe Creative Suite"],
            "description": "Design user experiences for web and mobile applications, conducting user research and testing. Strong portfolio required.",
            "experience_level": ["Entry", "Mid", "Senior"],
            "location": ["San Francisco", "New York", "Seattle", "Austin", "Remote"],
            "salary_range": (70000, 160000)
        },
        "Marketing Specialist": {
            "skills": ["Digital Marketing", "SEO", "Google Analytics", "Social Media", "Content Creation", "Campaign Management"],
            "description": "Create marketing campaigns, analyze market trends, and optimize digital marketing strategies. Data-driven approach required.",
            "experience_level": ["Entry", "Mid", "Senior"],
            "location": ["New York", "Los Angeles", "Chicago", "Austin", "Remote"],
            "salary_range": (50000, 120000)
        },
        "DevOps Engineer": {
            "skills": ["AWS", "Docker", "Kubernetes", "CI/CD", "Python", "Linux", "Terraform"],
            "description": "Manage infrastructure, automate deployments, and ensure system reliability. Experience with cloud platforms required.",
            "experience_level": ["Mid", "Senior", "Lead"],
            "location": ["San Francisco", "Seattle", "Austin", "Remote"],
            "salary_range": (90000, 190000)
        },
        "Data Engineer": {
            "skills": ["Python", "SQL", "Spark", "Airflow", "AWS", "ETL", "Data Pipeline"],
            "description": "Build and maintain data pipelines, optimize data processing systems. Strong background in distributed systems.",
            "experience_level": ["Mid", "Senior", "Lead"],
            "location": ["San Francisco", "New York", "Seattle", "Remote"],
            "salary_range": (95000, 200000)
        },
        "Machine Learning Engineer": {
            "skills": ["Python", "TensorFlow", "PyTorch", "MLOps", "AWS", "Docker", "Statistics"],
            "description": "Design and implement machine learning systems, deploy models to production. Strong background in ML algorithms.",
            "experience_level": ["Mid", "Senior", "Lead"],
            "location": ["San Francisco", "Seattle", "New York", "Remote"],
            "salary_range": (100000, 220000)
        }
    }
    
    jobs = []
    job_titles = list(job_templates.keys())
    
    for i in range(n_jobs):
        title = random.choice(job_titles)
        template = job_templates[title]
        
        # Generate job with some variation
        skills = random.sample(template["skills"], k=random.randint(3, len(template["skills"])))
        experience = random.choice(template["experience_level"])
        location = random.choice(template["location"])
        
        # Add some salary variation based on experience and location
        base_salary = random.randint(*template["salary_range"])
        if experience == "Senior" or experience == "Lead":
            base_salary = int(base_salary * random.uniform(1.1, 1.3))
        if location in ["San Francisco", "New York"]:
            base_salary = int(base_salary * random.uniform(1.1, 1.2))
            
        job = {
            "job_id": f"job_{i:04d}",
            "title": title,
            "description": template["description"],
            "required_skills": ", ".join(skills),
            "experience_level": experience,
            "location": location,
            "salary_min": base_salary,
            "salary_max": int(base_salary * random.uniform(1.1, 1.4)),
            "company": f"Company_{random.randint(1, 50):02d}",
            "posted_date": datetime.now() - timedelta(days=random.randint(1, 90)),
            "job_type": random.choice(["Full-time", "Part-time", "Contract", "Internship"]),
            "remote_allowed": random.choice([True, False]) if location != "Remote" else True
        }
        jobs.append(job)
    
    df = pd.DataFrame(jobs)
    
    if output_path:
        df.to_csv(output_path, index=False)
        
    return df


def generate_user_data(n_users: int = 500, output_path: Optional[str] = None) -> pd.DataFrame:
    """Generate realistic user profile data.
    
    Args:
        n_users: Number of users to generate
        output_path: Optional path to save the data
        
    Returns:
        DataFrame with user information
    """
    set_seed(42)
    
    # User skill profiles
    skill_categories = {
        "Technical": ["Python", "Java", "JavaScript", "React", "Node.js", "SQL", "Git", "AWS", "Docker", "Kubernetes"],
        "Data": ["Python", "R", "SQL", "Machine Learning", "Statistics", "Pandas", "Scikit-learn", "TensorFlow", "PyTorch"],
        "Design": ["Figma", "Sketch", "User Research", "Prototyping", "Design Systems", "Adobe Creative Suite"],
        "Product": ["Product Strategy", "Agile", "Analytics", "User Research", "Leadership", "Communication"],
        "Marketing": ["Digital Marketing", "SEO", "Google Analytics", "Social Media", "Content Creation", "Campaign Management"]
    }
    
    users = []
    
    for i in range(n_users):
        # Choose primary skill category
        primary_category = random.choice(list(skill_categories.keys()))
        primary_skills = random.sample(skill_categories[primary_category], k=random.randint(2, 4))
        
        # Add some secondary skills
        secondary_skills = []
        for category, skills in skill_categories.items():
            if category != primary_category:
                secondary_skills.extend(random.sample(skills, k=random.randint(0, 2)))
        
        all_skills = primary_skills + secondary_skills[:random.randint(1, 3)]
        
        user = {
            "user_id": f"user_{i:04d}",
            "name": f"User_{i:04d}",
            "skills": ", ".join(all_skills),
            "experience_years": random.randint(0, 15),
            "education_level": random.choice(["Bachelor's", "Master's", "PhD", "High School", "Bootcamp"]),
            "location": random.choice(["San Francisco", "New York", "Seattle", "Austin", "Boston", "Chicago", "Remote"]),
            "preferred_location": random.choice(["San Francisco", "New York", "Seattle", "Austin", "Remote"]),
            "salary_expectation": random.randint(50000, 200000),
            "job_type_preference": random.choice(["Full-time", "Part-time", "Contract", "Internship"]),
            "remote_preference": random.choice([True, False])
        }
        users.append(user)
    
    df = pd.DataFrame(users)
    
    if output_path:
        df.to_csv(output_path, index=False)
        
    return df


def generate_interactions(jobs_df: pd.DataFrame, users_df: pd.DataFrame, 
                         n_interactions: int = 5000, output_path: Optional[str] = None) -> pd.DataFrame:
    """Generate realistic user-job interactions.
    
    Args:
        jobs_df: DataFrame with job information
        users_df: DataFrame with user information
        n_interactions: Number of interactions to generate
        output_path: Optional path to save the data
        
    Returns:
        DataFrame with interaction data
    """
    set_seed(42)
    
    interactions = []
    
    for _ in range(n_interactions):
        user_id = random.choice(users_df["user_id"].values)
        job_id = random.choice(jobs_df["job_id"].values)
        
        # Get user and job data
        user = users_df[users_df["user_id"] == user_id].iloc[0]
        job = jobs_df[jobs_df["job_id"] == job_id].iloc[0]
        
        # Calculate compatibility score based on skills match
        user_skills = set(user["skills"].lower().split(", "))
        job_skills = set(job["required_skills"].lower().split(", "))
        skill_match = len(user_skills.intersection(job_skills)) / len(job_skills) if job_skills else 0
        
        # Calculate location compatibility
        location_match = 1.0 if user["preferred_location"] == job["location"] else 0.5
        
        # Calculate salary compatibility
        salary_match = 1.0 if user["salary_expectation"] <= job["salary_max"] else 0.3
        
        # Calculate overall compatibility
        compatibility = (skill_match * 0.5 + location_match * 0.3 + salary_match * 0.2)
        
        # Generate interaction based on compatibility
        if compatibility > 0.7:
            interaction_type = random.choice(["apply", "view", "save"])
            weight = random.uniform(0.8, 1.0)
        elif compatibility > 0.4:
            interaction_type = random.choice(["view", "save"])
            weight = random.uniform(0.4, 0.7)
        else:
            interaction_type = "view"
            weight = random.uniform(0.1, 0.4)
        
        # Add some randomness
        if random.random() < 0.1:  # 10% chance of random interaction
            interaction_type = random.choice(["apply", "view", "save"])
            weight = random.uniform(0.1, 1.0)
        
        interaction = {
            "user_id": user_id,
            "job_id": job_id,
            "interaction_type": interaction_type,
            "weight": weight,
            "timestamp": datetime.now() - timedelta(days=random.randint(1, 180)),
            "compatibility_score": compatibility
        }
        interactions.append(interaction)
    
    df = pd.DataFrame(interactions)
    
    if output_path:
        df.to_csv(output_path, index=False)
        
    return df


def load_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load or generate job recommendation data.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (jobs_df, users_df, interactions_df)
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    jobs_path = data_path / "jobs.csv"
    users_path = data_path / "users.csv"
    interactions_path = data_path / "interactions.csv"
    
    # Generate data if it doesn't exist
    if not jobs_path.exists():
        print("Generating job data...")
        jobs_df = generate_job_data(output_path=str(jobs_path))
    else:
        jobs_df = pd.read_csv(jobs_path)
    
    if not users_path.exists():
        print("Generating user data...")
        users_df = generate_user_data(output_path=str(users_path))
    else:
        users_df = pd.read_csv(users_path)
    
    if not interactions_path.exists():
        print("Generating interaction data...")
        interactions_df = generate_interactions(jobs_df, users_df, output_path=str(interactions_path))
    else:
        interactions_df = pd.read_csv(interactions_path)
    
    return jobs_df, users_df, interactions_df


if __name__ == "__main__":
    # Generate sample data
    jobs_df, users_df, interactions_df = load_data()
    print(f"Generated {len(jobs_df)} jobs, {len(users_df)} users, {len(interactions_df)} interactions")
    print("\nJobs sample:")
    print(jobs_df.head())
    print("\nUsers sample:")
    print(users_df.head())
    print("\nInteractions sample:")
    print(interactions_df.head())
