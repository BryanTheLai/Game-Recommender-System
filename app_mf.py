# app_mf.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
models_dir = os.path.join(BASE_DIR, "Game-recommender-system\\models")

# --- Load Artifacts ---
@st.cache_data
def load_artifacts():
    item_factors_path = os.path.join(models_dir, "item_factors.npy")
    item_ids_path = os.path.join(models_dir, "item_ids.pkl")
    games_metadata_path = os.path.join(BASE_DIR, "Game-recommender-system/data/external/games_tagged.csv")

    item_factors = np.load(item_factors_path)
    with open(item_ids_path, 'rb') as f:
        item_ids = pickle.load(f)
    games_df = pd.read_csv(games_metadata_path)

    return item_factors, item_ids, games_df

item_factors, item_ids, games_df = load_artifacts()

# Create mappings
app_id_to_index = {app_id: idx for idx, app_id in enumerate(item_ids)}
index_to_app_id = {idx: app_id for idx, app_id in enumerate(item_ids)}

# --- App Title ---
st.title("🎮 Game Recommendation System - Matrix Factorization Version")

st.markdown("""
This system recommends similar games based on hidden patterns learned from user-game interactions using Matrix Factorization (TruncatedSVD).
""")

# --- User Input ---
st.subheader("🔍 Select a Game")

available_titles = games_df[games_df['app_id'].isin(item_ids)].sort_values('title')['title'].tolist()
selected_game_title = st.selectbox("Choose a game title:", available_titles)

top_n = st.slider("How many recommendations do you want?", 5, 20, 10)

# --- Recommend Function ---
def recommend_similar_games(game_title, top_n=10):
    match = games_df[games_df['title'].str.lower() == game_title.lower()]
    if match.empty:
        return pd.DataFrame()

    app_id = match['app_id'].values[0]
    if app_id not in app_id_to_index:
        return pd.DataFrame()

    idx = app_id_to_index[app_id]
    target_vector = item_factors[idx].reshape(1, -1)
    similarities = cosine_similarity(target_vector, item_factors)[0]
    similar_indices = similarities.argsort()[::-1][1:top_n+1]

    recommendations = []
    for sim_idx in similar_indices:
        similar_app_id = index_to_app_id[sim_idx]
        game_row = games_df[games_df['app_id'] == similar_app_id]
        similar_title = game_row['title'].values[0] if not game_row.empty else "Unknown Title"
        sim_score = similarities[sim_idx]
        recommendations.append((similar_app_id, similar_title, sim_score))

    return pd.DataFrame(recommendations, columns=["App ID", "Game Title", "Similarity Score"])

# --- Recommend and Display ---
if selected_game_title:
    st.subheader(f"🎯 Top {top_n} Games Similar to '{selected_game_title}'")
    results_df = recommend_similar_games(selected_game_title, top_n=top_n)

    if not results_df.empty:
        st.dataframe(results_df.style.format({"Similarity Score": "{:.4f}"}))
    else:
        st.warning("No recommendations found. Try another game.")

# --- Footer ---
st.markdown("---")
st.caption("Developed based on Matrix Factorization (TruncatedSVD).")
