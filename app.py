# app.py
import streamlit as st
import pandas as pd
import numpy as np # Although not used directly in this snippet, cosine_similarity matrix is numpy
import pickle
import os
from typing import Dict, List, Tuple # For type hinting

# --- Configuration & Constants ---
MODELS_DIR = "models" # Relative path to the models directory
SIMILARITY_MATRIX_FILENAME = "item_similarity_matrix.pkl"
ITEM_MAP_FILENAME = "item_map.pkl"
ITEM_MAP_INV_FILENAME = "item_map_inv.pkl"
GAMES_DF_FILENAME = "filtered_games_metadata.pkl"

# --- Caching Data Loading ---
# Use st.cache_data to load these artifacts only once
@st.cache_data
def load_artifacts() -> Tuple[np.ndarray, Dict[int, int], Dict[int, int], pd.DataFrame]:
    """Loads the recommender artifacts from pickle files."""
    print("Loading artifacts...") # Add print statement to see when cache is missed
    artifacts_loaded = {}
    required_files = {
        'similarity_matrix': os.path.join(MODELS_DIR, SIMILARITY_MATRIX_FILENAME),
        'item_map': os.path.join(MODELS_DIR, ITEM_MAP_FILENAME),
        'item_map_inv': os.path.join(MODELS_DIR, ITEM_MAP_INV_FILENAME),
        'games_df': os.path.join(MODELS_DIR, GAMES_DF_FILENAME),
    }

    for name, path in required_files.items():
        try:
            with open(path, 'rb') as f:
                artifacts_loaded[name] = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Error: Artifact file not found at {path}. Make sure the model artifacts are in the '{MODELS_DIR}' directory.")
            return None, None, None, None # Return None for all if one is missing
        except Exception as e:
            st.error(f"Error loading artifact {name} from {path}: {e}")
            return None, None, None, None

    print("Artifacts loaded successfully.")
    return (
        artifacts_loaded['similarity_matrix'],
        artifacts_loaded['item_map'],
        artifacts_loaded['item_map_inv'],
        artifacts_loaded['games_df']
    )

# --- Recommendation Logic ---
# (Adapted from notebook Cell 8 - Takes loaded artifacts as input)
def recommend_similar_games_cosine(
    target_app_id: int,
    similarity_matrix: np.ndarray,
    item_map: Dict[int, int],
    item_map_inv: Dict[int, int],
    games_df: pd.DataFrame,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Recommends games similar to the target_app_id using loaded artifacts.
    """
    recommendations = []
    if target_app_id not in item_map:
        # This case should ideally be prevented by UI, but good to have
        st.warning(f"Selected game ID {target_app_id} not found in model data.")
        return pd.DataFrame(columns=['app_id', 'game_name', 'similarity_score'])

    try:
        item_index = item_map[target_app_id]
        similarity_scores = similarity_matrix[item_index]
        item_score_pairs = list(enumerate(similarity_scores))
        sorted_item_score_pairs = sorted(item_score_pairs, key=lambda x: x[1], reverse=True)
        top_similar_items = sorted_item_score_pairs[1 : top_n + 1]

        for matrix_idx, score in top_similar_items:
            try:
                recommended_app_id = item_map_inv[matrix_idx]
                game_info = games_df[games_df['app_id'] == recommended_app_id]
                game_name = game_info['title'].iloc[0] if not game_info.empty else "Title Not Found"

                # Ensure score is a standard Python float for display compatibility
                display_score = float(score)

                recommendations.append({
                    'app_id': recommended_app_id,
                    'game_name': game_name,
                    # Format score for better readability
                    'similarity_score': f"{display_score:.4f}"
                })
            except (KeyError, IndexError) as e:
                 print(f"Warning: Error processing recommendation index {matrix_idx}: {e}") # Log to console

    except Exception as e:
        st.error(f"An error occurred during recommendation generation: {e}")
        return pd.DataFrame(columns=['app_id', 'game_name', 'similarity_score'])

    return pd.DataFrame(recommendations)


# --- Streamlit App Layout ---

st.set_page_config(layout="wide") # Use wider layout

# Load artifacts, proceed only if successful
item_similarity_matrix, item_map, item_map_inv, games_pd = load_artifacts()

if item_similarity_matrix is None:
    st.stop() # Stop execution if artifacts failed to load

# --- App Title and Explanation ---
st.title("🎮 Game Recommender: Item-Based Collaborative Filtering")
st.markdown("""
This app recommends games based on **collaborative filtering**. It analyzes patterns in user interaction data
(like playtime or reviews) to find games that are frequently enjoyed by the *same groups of people*.

Select a game you like below, and the app will suggest others that players with similar tastes also played.
The **similarity score** indicates how closely related the games are based on shared user behavior.
""")
st.divider()

# --- User Input Area ---
col1, col2 = st.columns([2, 1]) # Create columns for layout

with col1:
    st.subheader("Select a Game")

    # --- Optional Tag Filtering ---
    # Extract unique tags (handle potential list-like strings)
    try:
        all_tags = set()
        # Safely evaluate the string representation of lists in the 'tags' column
        games_pd['tags_list'] = games_pd['tags'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])
        for tags in games_pd['tags_list']:
            if isinstance(tags, list):
                all_tags.update(tags)
        valid_tags = sorted(list(all_tags))
    except Exception as e:
        st.warning(f"Could not process tags for filtering: {e}")
        valid_tags = []

    if valid_tags:
        selected_tags = st.multiselect(
            "Filter games by tags (optional):",
            options=valid_tags,
            # default=[] # No default tags selected
        )
    else:
        selected_tags = [] # Ensure selected_tags exists even if tag processing fails

    # Filter games based on selected tags
    if selected_tags:
        # Keep games that have ALL selected tags
        filtered_games_list = games_pd[
            games_pd['tags_list'].apply(lambda game_tags: all(tag in game_tags for tag in selected_tags))
        ]
    else:
        # If no tags are selected, show all games
        filtered_games_list = games_pd

    # Prepare game titles for selectbox (handle potential duplicates if any)
    # Sort by title for easier searching
    selectable_games = filtered_games_list[['app_id', 'title']].drop_duplicates().sort_values('title')

    # --- Game Selection Dropdown ---
    selected_title = st.selectbox(
        "Select a game you like:",
        options=selectable_games['title'],
        index=None, # No default selection
        placeholder="Choose a game...",
    )

with col2:
    st.subheader("Settings")
    # --- Number of Recommendations Slider ---
    top_n = st.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=25,
        value=10, # Default value
        step=1
    )

st.divider()

# --- Display Recommendations ---
if selected_title:
    try:
        # Find the app_id for the selected title
        selected_game_info = selectable_games[selectable_games['title'] == selected_title]

        if not selected_game_info.empty:
            selected_app_id = selected_game_info['app_id'].iloc[0]

            st.subheader(f"Recommendations similar to '{selected_title}' (ID: {selected_app_id})")

            # Generate recommendations
            recommendations_df = recommend_similar_games_cosine(
                target_app_id=selected_app_id,
                similarity_matrix=item_similarity_matrix,
                item_map=item_map,
                item_map_inv=item_map_inv,
                games_df=games_pd, # Pass the full games df for lookup
                top_n=top_n
            )

            if not recommendations_df.empty:
                # Display recommendations in a table
                st.dataframe(
                    recommendations_df,
                    column_config={
                        "app_id": st.column_config.NumberColumn("App ID", format="%d"),
                        "game_name": st.column_config.TextColumn("Recommended Game Title"),
                        "similarity_score": st.column_config.NumberColumn("Similarity Score", format="%.4f"),
                    },
                    hide_index=True,
                    use_container_width=True
                )

                # Add the crucial explanation
                st.markdown(f"_*These recommendations are based on collaborative filtering: games frequently played/reviewed by the same users who engaged with '{selected_title}'._")
            else:
                st.info("No similar games found based on the current filtering and model data.")
        else:
            st.error("Could not find the selected game's App ID.") # Should not happen with selectbox

    except Exception as e:
        st.error(f"An error occurred while generating recommendations for '{selected_title}': {e}")
        print(f"Error details: {e}") # Log detailed error to console

elif st.session_state.get('ran_once', False): # Only show if not the very first run
     st.info("Select a game from the dropdown above to see recommendations.")

# Mark that the app has run at least once
st.session_state['ran_once'] = True