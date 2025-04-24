# app_cont.py (or app_content_based.py)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Set
import scipy.sparse # For type hint

# --- !!! MOVE THIS HERE - Must be the first Streamlit command !!! ---
st.set_page_config(layout="wide", page_title="Game Recommender (Content-Based)")
# ---------------------------------------------------------------------

# --- Configuration & Constants ---
MODELS_DIR = "models"
# Artifacts specific to the Content-Based Filtering model (matching notebook Cell 9)
NN_MODEL_FILENAME = "content_nn_model.pkl"
TFIDF_MATRIX_FILENAME = "content_tfidf_matrix.pkl"
ITEM_MAP_FILENAME = "content_item_map.pkl"
ITEM_MAP_INV_FILENAME = "content_item_map_inv.pkl"
GAMES_DF_FILENAME = "content_filtered_games_metadata.pkl"

# --- Caching Data Loading ---
@st.cache_data
def load_cbf_artifacts() -> Tuple[object | None, scipy.sparse.csr_matrix | None, Dict[int, int] | None, Dict[int, int] | None, pd.DataFrame | None]:
    """Loads Content-Based Filtering artifacts using specific filenames."""
    print("Attempting to load Content-Based Filtering artifacts...")
    artifacts = {}
    required_files = {
        'nn_model': os.path.join(MODELS_DIR, NN_MODEL_FILENAME),
        'tfidf_matrix': os.path.join(MODELS_DIR, TFIDF_MATRIX_FILENAME),
        'item_map': os.path.join(MODELS_DIR, ITEM_MAP_FILENAME),
        'item_map_inv': os.path.join(MODELS_DIR, ITEM_MAP_INV_FILENAME),
        'games_df': os.path.join(MODELS_DIR, GAMES_DF_FILENAME),
    }
    success = True
    for name, path in required_files.items():
        if not os.path.exists(path):
            st.error(f"Error: CBF Artifact '{path}' not found.") # Use st.error here
            success = False
            artifacts[name] = None
        else:
            try:
                with open(path, 'rb') as f:
                    artifacts[name] = pickle.load(f)
            except Exception as e:
                st.error(f"Error loading CBF artifact '{name}': {e}") # Use st.error here
                success = False
                artifacts[name] = None
    if not success:
        print("One or more CBF artifacts failed to load.")
        # Check happens in main_cbf
    print("All CBF artifacts loaded successfully.")
    return tuple(artifacts.values()) # Return in defined order

# --- CBF Recommendation Logic ---
def recommend_for_set_content(
    target_app_ids: List[int],
    nn_model: object, # Type hint as object, actual type is NearestNeighbors
    tfidf_matrix: scipy.sparse.csr_matrix,
    item_map: Dict[int, int],
    item_map_inv: Dict[int, int],
    games_df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """Recommends games based on content similarity using KNN on TF-IDF vectors."""
    if not target_app_ids:
        return pd.DataFrame(columns=['app_id', 'game_name', 'content_similarity'])

    # Get valid indices for input games
    input_indices: List[int] = []
    valid_input_app_ids: Set[int] = set()
    for app_id in target_app_ids:
        if app_id in item_map:
            idx = item_map[app_id]
            if 0 <= idx < tfidf_matrix.shape[0]:
                input_indices.append(idx)
                valid_input_app_ids.add(app_id)

    if not input_indices:
        print("Warning (CBF): None of the selected app_ids found in item_map.")
        return pd.DataFrame(columns=['app_id', 'game_name', 'content_similarity'])

    # Calculate average profile vector for the input games
    profile_vector = tfidf_matrix[input_indices].mean(axis=0)
    if isinstance(profile_vector, np.matrix): # Convert matrix to ndarray if necessary
        profile_vector = profile_vector.A
    if np.count_nonzero(profile_vector) == 0:
        print("Warning (CBF): Calculated profile vector is zero.")
        return pd.DataFrame(columns=['app_id', 'game_name', 'content_similarity'])

    # Find nearest neighbors using the loaded KNN model
    n_fetch = top_n + len(input_indices) + 10 # Fetch extra to filter inputs
    n_fetch = min(n_fetch, tfidf_matrix.shape[0])
    try:
        # nn_model expects 2D input
        distances, indices = nn_model.kneighbors(profile_vector.reshape(1, -1), n_neighbors=n_fetch)
    except Exception as e:
        st.error(f"Error during KNN search: {e}")
        return pd.DataFrame(columns=['app_id', 'game_name', 'content_similarity'])

    # Process neighbors: filter inputs, map back to app_id/name, calculate similarity
    recommendations = []
    for i, matrix_idx in enumerate(indices[0]): # Results are in the first row
        rec_app_id = item_map_inv.get(matrix_idx) # Use .get for safety
        if rec_app_id is not None and rec_app_id not in valid_input_app_ids:
            try:
                # Use .loc for potentially faster lookup if index is aligned (as done in notebook)
                game_info = games_df.loc[matrix_idx]
                # Convert distance to similarity (Cosine Similarity = 1 - Cosine Distance)
                similarity = 1.0 - distances[0][i]
                recommendations.append({
                    'app_id': rec_app_id,
                    'game_name': game_info['title'] if pd.notna(game_info['title']) else "N/A",
                    'content_similarity': float(similarity)
                })
                if len(recommendations) >= top_n:
                    break # Stop once we have enough
            except Exception as e: # Catch potential KeyErrors or IndexErrors during lookup
                print(f"Warning (CBF): Error processing recommended index {matrix_idx}: {e}")

    return pd.DataFrame(recommendations)

# --- Streamlit App Main Function ---
def main_cbf():
    # Load artifacts safely
    loaded_artifacts = load_cbf_artifacts()
    # Check for None AFTER loading
    if any(a is None for a in loaded_artifacts):
        # Errors already shown
        st.warning("Cannot proceed due to missing/failed artifacts.")
        st.stop()
    nn_model, tfidf_matrix, item_map, item_map_inv, games_pd = loaded_artifacts

    # Prepare games list for UI selector (only games known to the model)
    selectable_games_list = [
        {'app_id': app_id, 'title': title}
        for app_id, title in games_pd[['app_id', 'title']].values
        if app_id in item_map and pd.notna(title) and title.strip()
    ]
    if not selectable_games_list:
        st.error("No selectable games found matching the CBF model's vocabulary.")
        st.stop()
    selectable_games_df = pd.DataFrame(selectable_games_list).drop_duplicates().sort_values('title')

    # --- UI Elements ---
    st.title("🎮 Steam Recommender (Content-Based Filtering)")
    st.markdown("Select games you like to get recommendations based on **game content** (titles, tags, descriptions).")
    st.divider()

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_titles = st.multiselect(
            "Select Game(s) You Like:",
            options=selectable_games_df['title'].tolist(),
            placeholder="Type or select game titles...",
            label_visibility="collapsed"
        )
    with col2:
        top_n = st.slider("Number of recommendations:", 5, 25, 10, key="top_n_slider_cbf")
    st.divider()

    # --- Recommendation Display ---
    if selected_titles:
        selected_app_ids = selectable_games_df[selectable_games_df['title'].isin(selected_titles)]['app_id'].tolist()
        if selected_app_ids:
            st.subheader(f"Recommendations based on: {', '.join(f'{t}' for t in selected_titles)}")
            try:
                recommendations_df = recommend_for_set_content(
                    selected_app_ids, nn_model, tfidf_matrix, item_map, item_map_inv, games_pd, top_n
                )
                if not recommendations_df.empty:
                    st.dataframe(
                        recommendations_df,
                        column_config={
                            "app_id": st.column_config.NumberColumn("App ID"),
                            "game_name": st.column_config.TextColumn("Recommended Game"),
                            "content_similarity": st.column_config.NumberColumn(
                                "Content Similarity", help="Higher score (closer to 1) = More similar content.", format="%.4f"
                            ),
                        }, hide_index=True, use_container_width=True
                    )
                else:
                    st.info("No suitable content-based recommendations found.")
            except Exception as e:
                st.error(f"Error generating CBF recommendations: {e}")
                print(f"CBF Rec Error: {e}") # Log to console
        else:
            st.warning("Could not find App IDs for selected titles.")

    # Show initial message only after first interaction if nothing is selected
    if 'cbf_app_ran_once' not in st.session_state: st.session_state.cbf_app_ran_once = False
    if not selected_titles and st.session_state.cbf_app_ran_once: st.info("Select games above.")
    st.session_state.cbf_app_ran_once = True

# --- Run the App ---
if __name__ == "__main__":
    main_cbf()