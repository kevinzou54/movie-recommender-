import sys
import os
import numpy as np
from rapidfuzz import process

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from app.preprocess import (
    load_data,
    preprocess_movies,
    save_preprocessed_data,
    load_preprocessed_data,
)
from app.recommender import Recommender


def fuzzy_match_movies(input_movies, all_titles, threshold=90):
    """
    Match user-provided movie titles to the dataset using fuzzy matching.
    Args:
        input_movies (list): List of movies provided by the user.
        all_titles (list): List of all movie titles in the dataset.
        threshold (int): Minimum match score to consider a title as valid.
    Returns:
        matched_movies (list): List of matched movie titles.
    """
    matched_movies = []
    for movie in input_movies:
        matches = process.extract(movie, all_titles, limit=3)
        valid_matches = [match for match, score, _ in matches if score >= threshold]

        if not valid_matches:
            print(f"Could not find a close match for '{movie}'.")
        elif len(valid_matches) == 1:
            matched_movies.append(valid_matches[0])
        else:
            print(f"Multiple matches found for '{movie}': {valid_matches}")
            chosen_match = input(f"Select the best match for '{movie}' or press Enter to skip: ")
            if chosen_match in valid_matches:
                matched_movies.append(chosen_match)

    return matched_movies


if __name__ == "__main__":
    # Check for preprocessed data
    movies, ratings = load_preprocessed_data()

    # If no preprocessed data exists, preprocess raw data and save as pickle files
    if movies is None or ratings is None:
        print("No preprocessed data found. Loading raw data...")
        movies, ratings = load_data()
        movies = preprocess_movies(movies)
        save_preprocessed_data(movies, ratings)

    # Initialize the recommender system
    recommender = Recommender(movies)

    # Try to load a trained model
    model_path = "models/recommender_model.pkl"
    recommender.load_model(model_path)

    # If no model is found, train a new one and save it
    if not recommender.model:
        print("No pre-trained model found. Training a new model...")
        # Example: Generate a dummy feature matrix for training
        feature_matrix = np.random.rand(len(movies), 10)
        recommender.train_model(feature_matrix)
        recommender.save_model(model_path)


    # Prompt the user for input
    print("Enter 5 movies you have watched and enjoyed:")
    input_movies = [input(f"Movie {i+1}: ") for i in range(5)]

    # Match user-provided titles to dataset titles
    all_titles = movies["title"].tolist()
    matched_movies = fuzzy_match_movies(input_movies, all_titles)

    if not matched_movies:
        print("No valid movies found after matching.")
    else:
        print(f"Movies found: {matched_movies}")

        # Filter the dataset for matched movies
        valid_movies = movies[movies["title"].isin(matched_movies)]

        # Generate recommendations
        recommendations = recommender.recommend_from_ratings(valid_movies, top_n=5)
        print("Recommendations based on your input:")
        print(recommendations)
