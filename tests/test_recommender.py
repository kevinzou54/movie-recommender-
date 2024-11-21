import sys
import os

# Add the root directory to Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pytest
from app.preprocess import load_data, preprocess_movies
from app.recommender import Recommender

def test_load_data():
    # Test that the data loads correctly
    movies, ratings = load_data()
    assert not movies.empty, "Movies dataset should not be empty"
    assert not ratings.empty, "Ratings dataset should not be empty"

def test_preprocess_movies():
    # Test that the movies dataset is properly preprocessed
    movies, _ = load_data()
    movies = preprocess_movies(movies)
    assert "genres" in movies.columns, "Genres column should exist in the movies dataset"

def test_recommendation():
    # Test the recommendation system
    movies, _ = load_data()
    movies = preprocess_movies(movies)
    recommender = Recommender(movies)
    recommendations = recommender.recommend("Toy Story (1995)", top_n=3)
    assert len(recommendations) == 3, "Should return 3 recommendations"
