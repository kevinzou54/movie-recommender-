# File: app/preprocess.py
import pandas as pd
import os

def load_data(ratings_path="data/ratings.csv", movies_path="data/movies.csv"):
    """
    Load the raw movies and ratings datasets from CSV.
    """
    ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating", "timestamp"])
    movies = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
    return movies, ratings

def preprocess_movies(movies):
    """
    Perform preprocessing on the movies dataset.
    """
    movies["genres"] = movies["genres"].fillna("").astype(str)
    return movies

def save_preprocessed_data(movies, ratings, movies_path="data/movies.pkl", ratings_path="data/ratings.pkl"):
    """
    Save preprocessed data to pickle files.
    """
    movies.to_pickle(movies_path)
    ratings.to_pickle(ratings_path)
    print(f"Preprocessed data saved to {movies_path} and {ratings_path}")

def load_preprocessed_data(movies_path="data/movies.pkl", ratings_path="data/ratings.pkl"):
    """
    Load preprocessed data from pickle files.
    """
    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        movies = pd.read_pickle(movies_path)
        ratings = pd.read_pickle(ratings_path)
        print("Preprocessed data loaded from pickle files")
        return movies, ratings
    else:
        print("Pickle files not found. Loading raw data...")
        return None, None
