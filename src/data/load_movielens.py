import pandas as pd
import os

def load_ml_1m(data_path):
    ratings = pd.read_csv(
        os.path.join(data_path, "ratings.dat"),
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    users = pd.read_csv(
        os.path.join(data_path, "users.dat"),
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"]
    )

    movies = pd.read_csv(
        os.path.join(data_path, "movies.dat"),
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movie_id", "title", "genres"]
    )

    return ratings, users, movies