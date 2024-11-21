import pandas as pd

def load_data(ratings_path="data/ratings.csv", movies_path="data/movies.csv"):
    """
    Load the movies and ratings datasets.
    Default paths are used for production, but custom paths can be provided for testing.
    """
    ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating", "timestamp"])
    movies = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
    return movies, ratings

def preprocess_movies(movies):
    """
    Preprocess the movies dataset by cleaning the genres column.
    Args:
        movies (DataFrame): Raw movies dataset.
    Returns:
        movies (DataFrame): Preprocessed movies dataset.
    """
    # Handle missing genres
    movies['genres'] = movies['genres'].fillna("")

    return movies

if __name__ == "__main__":
    # Test the preprocessing
    movies, ratings = load_data()
    movies = preprocess_movies(movies)
    print("Movies Data:")
    print(movies.head())
    print("Ratings Data:")
    print(ratings.head())
