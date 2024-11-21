import sys
import os

# Add the root directory to Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pytest
from app.preprocess import load_data, preprocess_movies
from app.recommender import Recommender

def test_load_data():
    # Use mock data paths for testing
    movies, ratings = load_data(
        ratings_path="tests/mock_data/ratings.csv",
        movies_path="tests/mock_data/movies.csv"
    )
    assert not movies.empty, "Movies dataset should not be empty"
    assert not ratings.empty, "Ratings dataset should not be empty"

def test_preprocess_movies():
    movies, _ = load_data(
        ratings_path="tests/mock_data/ratings.csv",
        movies_path="tests/mock_data/movies.csv"
    )
    movies = preprocess_movies(movies)
    assert "genres" in movies.columns, "Genres column should exist in the movies dataset"

def test_recommendation():
    movies, _ = load_data(
        ratings_path="tests/mock_data/ratings.csv",
        movies_path="tests/mock_data/movies.csv"
    )
    movies = preprocess_movies(movies)
    recommender = Recommender(movies)
    recommendations = recommender.recommend("Toy Story (1995)", top_n=3)
    assert len(recommendations) > 0, "Should return recommendations"

