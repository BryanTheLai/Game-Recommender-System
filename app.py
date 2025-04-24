# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple

# --- Configuration & Constants ---
MODELS_DIR = "models"  # Relative path to the models directory
SIMILARITY_MATRIX_FILENAME = "item_similarity_matrix.pkl"
ITEM_MAP_FILENAME = "item_map.pkl"
ITEM_MAP_INV_FILENAME = "item_map_inv.pkl"
GAMES_DF_FILENAME = "filtered_games_metadata.pkl"

# --- Caching Data Loading ---
@st.cache_data
def load_artifacts() -> Tuple[np.ndarray, Dict[int, int], Dict[int, int], pd.DataFrame]:
    """
    Loads the recommender artifacts (similarity matrix, item maps, games metadata)
    from pickle files using Streamlit's caching.
    """
    print("Attempting to load artifacts...")  # Log cache misses
    artifacts_loaded = {}
    required_files = {
        'similarity_matrix': os.path.join(MODELS_DIR, SIMILARITY_MATRIX_FILENAME),
        'item_map': os.path.join(MODELS_DIR, ITEM_MAP_FILENAME),
        'item_map_inv': os.path.join(MODELS_DIR, ITEM_MAP_INV_FILENAME),
        'games_df': os.path.join(MODELS_DIR, GAMES_DF_FILENAME),
    }

    all_loaded_successfully = True
    for name, path in required_files.items():
        if not os.path.exists(path):
            st.error(f"Error: Artifact file not found at '{path}'. Ensure it exists in the '{MODELS_DIR}' directory.")
            all_loaded_successfully = False
            artifacts_loaded[name] = None # Mark as None if missing
        else:
            try:
                with open(path, 'rb') as f:
                    artifacts_loaded[name] = pickle.load(f)
                print(f"Successfully loaded artifact: {name}")
            except Exception as e:
                st.error(f"Error loading artifact '{name}' from '{path}': {e}")
                all_loaded_successfully = False
                artifacts_loaded[name] = None # Mark as None if loading failed

    if not all_loaded_successfully:
        print("One or more artifacts failed to load.")
        # Return None for all if any failed, allows easier checking later
        return None, None, None, None

    print("All artifacts loaded successfully.")
    return (
        artifacts_loaded.get('similarity_matrix'),
        artifacts_loaded.get('item_map'),
        artifacts_loaded.get('item_map_inv'),
        artifacts_loaded.get('games_df')
    )

# --- Recommendation Logic ---
def recommend_for_set(
    target_app_ids: List[int],
    similarity_matrix: np.ndarray,
    item_map: Dict[int, int],
    item_map_inv: Dict[int, int],
    games_df: pd.DataFrame,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Recommends games based on a list of input game app_ids by aggregating
    their similarity scores. Excludes input games from recommendations.
    """
    recommendations = []
    if not target_app_ids:
        return pd.DataFrame(columns=['app_id', 'game_name', 'aggregated_score'])

    n_items = similarity_matrix.shape[0]
    aggregated_scores = np.zeros(n_items, dtype=np.float64) # Use float64 for potentially large sums
    input_indices = set()
    valid_input_count = 0

    # --- Aggregate scores from valid input games ---
    for app_id in target_app_ids:
        if app_id in item_map:
            item_index = item_map[app_id]
            # Safety check: Ensure index is within the matrix bounds
            if 0 <= item_index < n_items:
                aggregated_scores += similarity_matrix[item_index]
                input_indices.add(item_index)
                valid_input_count += 1
            else:
                # Log unexpected out-of-bounds index
                print(f"Debug Warning: app_id {app_id} mapped to out-of-bounds index {item_index}.")
        # Silently ignore app_ids not in item_map as they aren't part of the model

    if valid_input_count == 0:
        # This occurs if none of the selected games are in the model's item_map
        print("Warning: None of the selected app_ids were found in item_map.")
        return pd.DataFrame(columns=['app_id', 'game_name', 'aggregated_score'])

    # --- Generate candidate items (excluding input items) ---
    candidate_items = []
    for item_idx in range(n_items):
        # Only consider items that have an inverse mapping (are valid items in the model)
        # AND were NOT part of the input set
        if item_idx in item_map_inv and item_idx not in input_indices:
            candidate_items.append((item_idx, aggregated_scores[item_idx]))

    # --- Sort candidates by aggregated score ---
    candidate_items.sort(key=lambda x: x[1], reverse=True)

    # --- Get top N recommendations ---
    top_similar_items = candidate_items[:top_n]

    # --- Map Indices back to App IDs and Get Names ---
    for matrix_idx, score in top_similar_items:
        try:
            recommended_app_id = item_map_inv[matrix_idx]
            # Look up game title efficiently using .loc after setting app_id as index (optional optimization)
            # Or use the standard boolean indexing:
            game_info = games_df[games_df['app_id'] == recommended_app_id]
            if not game_info.empty:
                game_name = game_info['title'].iloc[0]
                recommendations.append({
                    'app_id': recommended_app_id,
                    'game_name': game_name,
                    'aggregated_score': float(score) # Ensure standard Python float
                })
            else:
                # Log if a recommended app_id doesn't have metadata (should be rare if games_df is filtered correctly)
                print(f"Warning: Metadata not found for recommended app_id {recommended_app_id}")
        except KeyError:
            # Log if an index from item_map_inv doesn't work (data inconsistency)
            print(f"Warning: Could not find app_id for matrix index {matrix_idx} in item_map_inv.")
        except IndexError:
             # Log if title lookup fails unexpectedly
             print(f"Warning: IndexError during title lookup for recommended app_id {recommended_app_id}.")

    return pd.DataFrame(recommendations)


# --- Streamlit App Main Function ---
def main():
    st.set_page_config(layout="wide", page_title="Game Recommender")

    # --- Load Artifacts ---
    item_similarity_matrix, item_map, item_map_inv, games_pd = load_artifacts()

    # --- Check if Artifacts Loaded ---
    if any(artifact is None for artifact in [item_similarity_matrix, item_map, item_map_inv, games_pd]):
        st.error("Failed to load one or more essential model artifacts. Cannot proceed.")
        st.stop() # Halt execution

    # --- Prepare Data for UI ---
    # Create a DataFrame of games known to the model for the selection widget
    # Ensure only games present in item_map (the model's vocabulary) are selectable
    selectable_games_list = []
    for app_id, title in games_pd[['app_id', 'title']].values:
        if app_id in item_map:
            selectable_games_list.append({'app_id': app_id, 'title': title})

    if not selectable_games_list:
        st.error("No selectable games found in the loaded metadata that are also present in the model's item map.")
        st.stop()

    selectable_games_df = pd.DataFrame(selectable_games_list).drop_duplicates().sort_values('title')

    # --- App Title and Explanation ---
    st.title("🎮 Steam Game Recommender")
    st.markdown("""
    Select **one or more games** you enjoy from the list below.
    This tool uses an item-based collaborative filtering model (based on user interaction patterns)
    to suggest other games you might like, based on the combined similarity to your selections.
    """)
    st.divider()

    # --- User Input Area ---
    col1, col2 = st.columns([3, 1]) # Give more space to selection

    with col1:
        st.subheader("Select Game(s) You Like")
        selected_titles = st.multiselect(
            label="Choose games from the list:", # Use label for clarity
            options=selectable_games_df['title'].tolist(), # Pass list of titles
            placeholder="Type or select game titles...",
            label_visibility="collapsed" # Hide label visually if subheader is enough
        )

    with col2:
        st.subheader("Settings")
        top_n = st.slider(
            "Number of recommendations:",
            min_value=5,
            max_value=25,
            value=10, # Default value
            step=1,
            key="top_n_slider" # Add key for stability
        )

    st.divider()

    # --- Generate and Display Recommendations ---
    if selected_titles: # Check if the list of selected titles is not empty
        try:
            # Get the app_ids for the selected titles
            selected_app_ids = selectable_games_df[selectable_games_df['title'].isin(selected_titles)]['app_id'].tolist()

            if selected_app_ids:
                titles_display = ", ".join([f"'{title}'" for title in selected_titles])
                st.subheader(f"Recommendations based on: {titles_display}")

                # Generate recommendations
                recommendations_df = recommend_for_set(
                    target_app_ids=selected_app_ids,
                    similarity_matrix=item_similarity_matrix,
                    item_map=item_map,
                    item_map_inv=item_map_inv,
                    games_df=games_pd,
                    top_n=top_n
                )

                if not recommendations_df.empty:
                    st.dataframe(
                        recommendations_df,
                        column_config={
                            "app_id": st.column_config.NumberColumn("App ID", format="%d"),
                            "game_name": st.column_config.TextColumn("Recommended Game"),
                            "aggregated_score": st.column_config.NumberColumn(
                                "Similarity Score",
                                help="Higher score means more similar to your selected set based on user interactions.",
                                format="%.4f"
                            ),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    st.caption(f"Displaying top {len(recommendations_df)} recommendations.")
                else:
                    # This might happen if valid games were selected but no suitable recommendations were found (e.g., all similar items were also selected)
                    st.info("No suitable recommendations found for the selected combination of games. Try different or fewer games.")
            else:
                 # Should be rare if multiselect options are correct
                 st.warning("Could not retrieve App IDs for selected titles.")

        except Exception as e:
            st.error("An unexpected error occurred while generating recommendations.")
            # Log the full error for debugging if needed (e.g., to console or a logging service)
            print(f"Error during recommendation generation: {e}")
            import traceback
            print(traceback.format_exc())

    # --- Initial State Message ---
    # Use session state to show message only after first interaction if nothing is selected
    if 'app_ran_once' not in st.session_state:
        st.session_state['app_ran_once'] = False # Initialize

    if not selected_titles and st.session_state['app_ran_once']:
        st.info("Select one or more games above to get recommendations.")

    # Mark that the app has run past the initial setup
    st.session_state['app_ran_once'] = True


# --- Run the App ---
if __name__ == "__main__":
    main()