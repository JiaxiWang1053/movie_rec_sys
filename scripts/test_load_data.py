from src.data.load_movielens import load_ml_1m

ratings, users, movies = load_ml_1m("data/raw/ml-1m")

print("Ratings:", ratings.shape)
print("Users:", users.shape)
print("Movies:", movies.shape)

print("\nSample ratings:")
print(ratings.head())