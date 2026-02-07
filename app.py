import streamlit as st
import numpy as np
import pandas as pd

# ---------------- LOAD DATA ----------------
movie_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv").head(10000)


model = np.load("recommender_model.npz")
X = model["X"]
W = model["W"]
b = model["b"]

# ---------------- BUILD MAPS ----------------
user_ids = ratings_df.userId.unique()
movie_ids = ratings_df.movieId.unique()

user_map = {u: i for i, u in enumerate(user_ids)}
movie_map = {m: i for i, m in enumerate(movie_ids)}
inv_movie_map = {v: k for k, v in movie_map.items()}

# ---------------- BUILD R MATRIX ----------------
nm, nu = len(movie_ids), len(user_ids)
R = np.zeros((nm, nu))

for row in ratings_df.itertuples():
    m = movie_map[row.movieId]
    u = user_map[row.userId]
    R[m, u] = 1

# ---------------- UI ----------------
st.title("🎬 Movie Recommendation System")
st.write("Collaborative Filtering based Recommender")

user_id = st.selectbox("Select a User ID", user_ids)

if st.button("Recommend Movies"):
    u = user_map[user_id]

    preds = X @ W[u] + b[0, u]
    preds[R[:, u] == 1] = -1e9

    top_idx = np.argsort(preds)[-10:][::-1]
    top_movie_ids = [inv_movie_map[i] for i in top_idx]

    recommendations = (
        movie_df[movie_df.movieId.isin(top_movie_ids)]
        .set_index("movieId")
        .loc[top_movie_ids]
        .reset_index()
    )

    st.subheader("Top 10 Recommended Movies")
    st.table(recommendations[["title"]])
