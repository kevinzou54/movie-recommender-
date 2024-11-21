from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import pickle

class Recommender:
    def __init__(self, movies):
        """
        Initialize the recommender with preprocessed movie data.
        Args:
            movies (DataFrame): Preprocessed movies dataset.
        """
        self.movies = movies
        self.tfidf_matrix = self._build_tfidf_matrix()
        self.model = None  # Ensure the model attribute is always initialized

    def _build_tfidf_matrix(self):
        """
        Build a TF-IDF matrix based on the genres of the movies.
        Returns:
            tfidf_matrix (sparse matrix): Matrix of TF-IDF features.
        """
        # Use TfidfVectorizer to encode genres
        tfidf = TfidfVectorizer(stop_words="english")
        return tfidf.fit_transform(self.movies["genres"])

    def train_model(self, feature_matrix):
        """
        Train a placeholder model. Replace with actual logic for collaborative filtering or other techniques.
        """
        from sklearn.neighbors import NearestNeighbors
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(feature_matrix)
        print("Model trained")

    def recommend(self, movie_title, top_n=5):
        """
        Recommend movies similar to the given title.
        Args:
            movie_title (str): The title of the movie to base recommendations on.
            top_n (int): Number of recommendations to return.
        Returns:
            recommendations (DataFrame): Top recommended movies.
        """
        # Check if the movie exists in the dataset
        if movie_title not in self.movies["title"].values:
            raise ValueError(f"Movie '{movie_title}' not found in the dataset.")

        # Find the index of the movie
        idx = self.movies[self.movies["title"] == movie_title].index[0]

        # Compute cosine similarity for the movie
        sim_scores = list(enumerate(cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0]))

        # Sort by similarity score (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N recommendations (skip the first one as it's the input movie itself)
        top_indices = [i[0] for i in sim_scores[1:top_n+1]]

        # Return the recommended movies
        return self.movies.iloc[top_indices][["title", "genres"]]
    
    def recommend_from_ratings(self, rated_movies, top_n=10):
        """
        Recommend movies based on a list of highly rated movies.
        Args:
            rated_movies (DataFrame): DataFrame of highly rated movies (user's preferences).
            top_n (int): Number of recommendations to return.
        Returns:
            recommendations (DataFrame): Top recommended movies.
        """
        all_sim_scores = {}

        # Iterate over each movie the user rated highly
        for _, row in rated_movies.iterrows():
            try:
                idx = self.movies[self.movies["title"] == row["title"]].index[0]
                sim_scores = list(enumerate(cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0]))
                for movie_idx, score in sim_scores:
                    if movie_idx not in all_sim_scores:
                        all_sim_scores[movie_idx] = 0
                    all_sim_scores[movie_idx] += score
            except IndexError:
                # Skip movies not found in the dataset
                continue

        # Sort aggregated scores
        sorted_movies = sorted(all_sim_scores.items(), key=lambda x: x[1], reverse=True)

        # Get top N recommended movies
        top_indices = [i[0] for i in sorted_movies[:top_n]]
        return self.movies.iloc[top_indices][["title", "genres"]]

    def save_model(self, model_path="models/recommender_model.pkl"):
        """
        Save the recommender object (including TF-IDF matrix) to a pickle file.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {model_path}")

    @staticmethod
    def load_model(model_path="models/recommender_model.pkl"):
        """
        Load the recommender object from a pickle file.
        """
        try:
            with open(model_path, "rb") as f:
                recommender = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return recommender
        except FileNotFoundError:
            print(f"Model file {model_path} not found.")
            return None
