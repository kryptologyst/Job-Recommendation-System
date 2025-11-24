"""Streamlit demo for job recommendation system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.job_recommender.data.generator import load_data, set_seed
from src.job_recommender.models.content_based import TFIDFRecommender, SentenceTransformerRecommender
from src.job_recommender.models.collaborative_filtering import (
    UserBasedCFRecommender, ItemBasedCFRecommender, MatrixFactorizationRecommender
)


@st.cache_data
def load_system_data():
    """Load data and models for the demo."""
    set_seed(42)
    
    # Load data
    jobs_df, users_df, interactions_df = load_data()
    
    # Train a simple TF-IDF model for demo
    tfidf_model = TFIDFRecommender()
    tfidf_model.fit(interactions_df, jobs_df, users_df)
    
    return jobs_df, users_df, interactions_df, tfidf_model


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Job Recommendation System",
        page_icon="💼",
        layout="wide"
    )
    
    st.title("💼 Job Recommendation System")
    st.markdown("A modern recommendation system for job matching using content-based and collaborative filtering approaches.")
    
    # Load data
    with st.spinner("Loading data and models..."):
        jobs_df, users_df, interactions_df, model = load_system_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "User Recommendations", "Job Similarity", "Data Analysis"]
    )
    
    if page == "Overview":
        show_overview(jobs_df, users_df, interactions_df)
    elif page == "User Recommendations":
        show_user_recommendations(jobs_df, users_df, interactions_df, model)
    elif page == "Job Similarity":
        show_job_similarity(jobs_df, model)
    elif page == "Data Analysis":
        show_data_analysis(jobs_df, users_df, interactions_df)


def show_overview(jobs_df, users_df, interactions_df):
    """Show system overview."""
    st.header("System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jobs", len(jobs_df))
    
    with col2:
        st.metric("Total Users", len(users_df))
    
    with col3:
        st.metric("Total Interactions", len(interactions_df))
    
    with col4:
        avg_interactions = len(interactions_df) / len(users_df) if len(users_df) > 0 else 0
        st.metric("Avg Interactions/User", f"{avg_interactions:.1f}")
    
    # Job distribution
    st.subheader("Job Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Job titles distribution
        title_counts = jobs_df['title'].value_counts().head(10)
        fig = px.bar(
            x=title_counts.values,
            y=title_counts.index,
            orientation='h',
            title="Top 10 Job Titles",
            labels={'x': 'Count', 'y': 'Job Title'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Location distribution
        location_counts = jobs_df['location'].value_counts()
        fig = px.pie(
            values=location_counts.values,
            names=location_counts.index,
            title="Job Distribution by Location"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Interaction patterns
    st.subheader("Interaction Patterns")
    
    # Interaction types
    interaction_counts = interactions_df['interaction_type'].value_counts()
    fig = px.bar(
        x=interaction_counts.index,
        y=interaction_counts.values,
        title="Interaction Types Distribution",
        labels={'x': 'Interaction Type', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # User activity
    user_activity = interactions_df.groupby('user_id').size().reset_index(name='interaction_count')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            user_activity,
            x='interaction_count',
            title="User Activity Distribution",
            labels={'interaction_count': 'Number of Interactions', 'count': 'Number of Users'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top active users
        top_users = user_activity.nlargest(10, 'interaction_count')
        fig = px.bar(
            top_users,
            x='user_id',
            y='interaction_count',
            title="Top 10 Most Active Users",
            labels={'interaction_count': 'Number of Interactions'}
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)


def show_user_recommendations(jobs_df, users_df, interactions_df, model):
    """Show user recommendation interface."""
    st.header("User Recommendations")
    
    # User selection
    user_id = st.selectbox(
        "Select a user",
        users_df['user_id'].tolist()
    )
    
    if user_id:
        # Show user profile
        user_info = users_df[users_df['user_id'] == user_id].iloc[0]
        
        st.subheader(f"User Profile: {user_info['name']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Skills:** {user_info['skills']}")
            st.write(f"**Experience:** {user_info['experience_years']} years")
        
        with col2:
            st.write(f"**Location:** {user_info['location']}")
            st.write(f"**Preferred Location:** {user_info['preferred_location']}")
        
        with col3:
            st.write(f"**Education:** {user_info['education_level']}")
            st.write(f"**Salary Expectation:** ${user_info['salary_expectation']:,}")
        
        # User's interaction history
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        if len(user_interactions) > 0:
            st.subheader("Interaction History")
            
            # Show interacted jobs
            interacted_jobs = user_interactions.merge(jobs_df, on='job_id')
            
            for _, interaction in interacted_jobs.iterrows():
                with st.expander(f"{interaction['title']} at {interaction['company']} ({interaction['interaction_type']})"):
                    st.write(f"**Description:** {interaction['description']}")
                    st.write(f"**Required Skills:** {interaction['required_skills']}")
                    st.write(f"**Location:** {interaction['location']}")
                    st.write(f"**Salary Range:** ${interaction['salary_min']:,} - ${interaction['salary_max']:,}")
                    st.write(f"**Interaction Weight:** {interaction['weight']:.2f}")
        
        # Generate recommendations
        n_recommendations = st.slider("Number of recommendations", 5, 20, 10)
        
        if st.button("Generate Recommendations"):
            with st.spinner("Generating recommendations..."):
                try:
                    recommendations = model.recommend(user_id, n_recommendations=n_recommendations)
                    
                    st.subheader("Recommended Jobs")
                    
                    for i, rec in enumerate(recommendations, 1):
                        job_info = jobs_df[jobs_df['job_id'] == rec['job_id']].iloc[0]
                        
                        with st.expander(f"{i}. {rec['title']} at {rec['company']} (Score: {rec['score']:.3f})"):
                            st.write(f"**Description:** {job_info['description']}")
                            st.write(f"**Required Skills:** {job_info['required_skills']}")
                            st.write(f"**Location:** {job_info['location']}")
                            st.write(f"**Experience Level:** {job_info['experience_level']}")
                            st.write(f"**Salary Range:** ${job_info['salary_min']:,} - ${job_info['salary_max']:,}")
                            st.write(f"**Job Type:** {job_info['job_type']}")
                            st.write(f"**Remote Allowed:** {'Yes' if job_info['remote_allowed'] else 'No'}")
                            
                            # Show why this job was recommended
                            st.write("**Why recommended:**")
                            st.write(f"- Content similarity score: {rec['score']:.3f}")
                            
                            # Check skill match
                            user_skills = set(user_info['skills'].lower().split(', '))
                            job_skills = set(job_info['required_skills'].lower().split(', '))
                            skill_match = len(user_skills.intersection(job_skills))
                            st.write(f"- Skill matches: {skill_match}/{len(job_skills)} required skills")
                            
                            # Check location match
                            location_match = user_info['preferred_location'] == job_info['location']
                            st.write(f"- Location preference match: {'Yes' if location_match else 'No'}")
                            
                            # Check salary compatibility
                            salary_compatible = user_info['salary_expectation'] <= job_info['salary_max']
                            st.write(f"- Salary compatible: {'Yes' if salary_compatible else 'No'}")
                
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")


def show_job_similarity(jobs_df, model):
    """Show job similarity interface."""
    st.header("Job Similarity Search")
    
    # Job selection
    job_id = st.selectbox(
        "Select a job to find similar ones",
        jobs_df['job_id'].tolist(),
        format_func=lambda x: f"{jobs_df[jobs_df['job_id'] == x].iloc[0]['title']} at {jobs_df[jobs_df['job_id'] == x].iloc[0]['company']}"
    )
    
    if job_id:
        job_info = jobs_df[jobs_df['job_id'] == job_id].iloc[0]
        
        st.subheader(f"Selected Job: {job_info['title']} at {job_info['company']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Description:** {job_info['description']}")
            st.write(f"**Required Skills:** {job_info['required_skills']}")
        
        with col2:
            st.write(f"**Location:** {job_info['location']}")
            st.write(f"**Experience Level:** {job_info['experience_level']}")
            st.write(f"**Salary Range:** ${job_info['salary_min']:,} - ${job_info['salary_max']:,}")
        
        # Find similar jobs
        n_similar = st.slider("Number of similar jobs", 5, 15, 10)
        
        if st.button("Find Similar Jobs"):
            with st.spinner("Finding similar jobs..."):
                try:
                    similar_jobs = model.get_similar_jobs(job_id, n_similar=n_similar)
                    
                    st.subheader("Similar Jobs")
                    
                    for i, similar in enumerate(similar_jobs, 1):
                        similar_job_info = jobs_df[jobs_df['job_id'] == similar['job_id']].iloc[0]
                        
                        with st.expander(f"{i}. {similar['title']} at {similar['company']} (Similarity: {similar['similarity']:.3f})"):
                            st.write(f"**Description:** {similar_job_info['description']}")
                            st.write(f"**Required Skills:** {similar_job_info['required_skills']}")
                            st.write(f"**Location:** {similar_job_info['location']}")
                            st.write(f"**Experience Level:** {similar_job_info['experience_level']}")
                            st.write(f"**Salary Range:** ${similar_job_info['salary_min']:,} - ${similar_job_info['salary_max']:,}")
                            st.write(f"**Job Type:** {similar_job_info['job_type']}")
                
                except Exception as e:
                    st.error(f"Error finding similar jobs: {str(e)}")


def show_data_analysis(jobs_df, users_df, interactions_df):
    """Show data analysis and insights."""
    st.header("Data Analysis & Insights")
    
    # Job market analysis
    st.subheader("Job Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary analysis
        salary_data = jobs_df[['title', 'salary_min', 'salary_max']].copy()
        salary_data['salary_avg'] = (salary_data['salary_min'] + salary_data['salary_max']) / 2
        
        fig = px.box(
            salary_data,
            x='title',
            y='salary_avg',
            title="Salary Distribution by Job Title",
            labels={'salary_avg': 'Average Salary ($)', 'title': 'Job Title'}
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Experience level distribution
        exp_counts = jobs_df['experience_level'].value_counts()
        fig = px.pie(
            values=exp_counts.values,
            names=exp_counts.index,
            title="Experience Level Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # User behavior analysis
    st.subheader("User Behavior Analysis")
    
    # Interaction patterns over time
    interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
    interactions_df['date'] = interactions_df['timestamp'].dt.date
    
    daily_interactions = interactions_df.groupby('date').size().reset_index(name='count')
    
    fig = px.line(
        daily_interactions,
        x='date',
        y='count',
        title="Daily Interaction Trends",
        labels={'count': 'Number of Interactions', 'date': 'Date'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Popular jobs analysis
    st.subheader("Popular Jobs Analysis")
    
    job_popularity = interactions_df['job_id'].value_counts().head(20)
    popular_jobs_df = pd.DataFrame({
        'job_id': job_popularity.index,
        'interaction_count': job_popularity.values
    }).merge(jobs_df, on='job_id')
    
    fig = px.bar(
        popular_jobs_df,
        x='interaction_count',
        y='title',
        orientation='h',
        title="Top 20 Most Popular Jobs",
        labels={'interaction_count': 'Number of Interactions', 'title': 'Job Title'}
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Skills analysis
    st.subheader("Skills Analysis")
    
    # Extract all skills
    all_skills = []
    for skills in jobs_df['required_skills']:
        all_skills.extend([skill.strip() for skill in skills.split(',')])
    
    skill_counts = pd.Series(all_skills).value_counts().head(20)
    
    fig = px.bar(
        x=skill_counts.values,
        y=skill_counts.index,
        orientation='h',
        title="Top 20 Most Required Skills",
        labels={'x': 'Frequency', 'y': 'Skill'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
