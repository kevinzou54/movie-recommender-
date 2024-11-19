import pandas as pd

def load_data():
    """
    Load the MovieLens 25M dataset from CSV files.
    Returns:
        movies (DataFrame): Movie metadata with titles and genres.
        ratings (DataFrame): User ratings for movies.
    """
    # Load ratings data
    ratings = pd.read_csv(
        'data/ratings.csv',  # Path to ratings.csv
        usecols=['userId', 'movieId', 'rating', 'timestamp']  # Only necessary columns
    )

    # Load movies data
    movies = pd.read_csv(
        'data/movies.csv',  # Path to movies.csv
        usecols=['movieId', 'title', 'genres']  # Only necessary columns
    )

    # Rename columns for consistency
    ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'}, inplace=True)
    movies.rename(columns={'movieId': 'movie_id'}, inplace=True)

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
