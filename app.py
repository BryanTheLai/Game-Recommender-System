# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Set, Any
import scipy.sparse  # For CBF type hint

# --- !!! Must be the first Streamlit command !!! ---
st.set_page_config(layout="wide", page_title="Steam Game Recommender")

# --- Configuration & Constants ---
MODELS_DIR = "models"

# CBF Artifact Filenames (Content-Based)
CBF_NN_MODEL_FILENAME = "content_nn_model.pkl"
CBF_TFIDF_MATRIX_FILENAME = "content_tfidf_matrix.pkl"
CBF_ITEM_MAP_FILENAME = "content_item_map.pkl"
CBF_ITEM_MAP_INV_FILENAME = "content_item_map_inv.pkl"
CBF_GAMES_DF_FILENAME = "content_filtered_games_metadata.pkl" # Specific games df for CBF

# CF Artifact Filenames (Collaborative Filtering)
CF_SIMILARITY_MATRIX_FILENAME = "item_similarity_matrix.pkl"
CF_ITEM_MAP_FILENAME = "item_map.pkl" # Potentially different item map for CF
CF_ITEM_MAP_INV_FILENAME = "item_map_inv.pkl" # Potentially different inverse map for CF
CF_GAMES_DF_FILENAME = "filtered_games_metadata.pkl" # Potentially different games df for CF

# --- Caching Data Loading ---

@st.cache_data
def load_cbf_artifacts() -> Tuple[Any | None, scipy.sparse.csr_matrix | None, Dict[int, int] | None, Dict[int, int] | None, pd.DataFrame | None]:
    """Loads Content-Based Filtering artifacts using specific filenames."""
    print("Attempting to load Content-Based Filtering artifacts...")
    artifacts = {}
    required_files = {
        'nn_model': os.path.join(MODELS_DIR, CBF_NN_MODEL_FILENAME),
        'tfidf_matrix': os.path.join(MODELS_DIR, CBF_TFIDF_MATRIX_FILENAME),
        'item_map': os.path.join(MODELS_DIR, CBF_ITEM_MAP_FILENAME),
        'item_map_inv': os.path.join(MODELS_DIR, CBF_ITEM_MAP_INV_FILENAME),
        'games_df': os.path.join(MODELS_DIR, CBF_GAMES_DF_FILENAME),
    }
    success = True
    for name, path in required_files.items():
        if not os.path.exists(path):
            st.error(f"Error: CBF Artifact '{path}' not found.")
            success = False
            artifacts[name] = None
        else:
            try:
                with open(path, 'rb') as f:
                    artifacts[name] = pickle.load(f)
                print(f"Successfully loaded CBF artifact: {name}")
            except Exception as e:
                st.error(f"Error loading CBF artifact '{name}': {e}")
                success = False
                artifacts[name] = None
    if not success:
        print("One or more CBF artifacts failed to load.")
        # Return None for all if any failed, allows easier checking later
        return None, None, None, None, None

    print("All CBF artifacts loaded successfully.")
    return (
        artifacts.get('nn_model'),
        artifacts.get('tfidf_matrix'),
        artifacts.get('item_map'),
        artifacts.get('item_map_inv'),
        artifacts.get('games_df')
    )

@st.cache_data
def load_cf_artifacts() -> Tuple[np.ndarray | None, Dict[int, int] | None, Dict[int, int] | None, pd.DataFrame | None]:
    """Loads Collaborative Filtering artifacts using specific filenames."""
    print("Attempting to load Collaborative Filtering artifacts...")
    artifacts_loaded = {}
    required_files = {
        'similarity_matrix': os.path.join(MODELS_DIR, CF_SIMILARITY_MATRIX_FILENAME),
        'item_map': os.path.join(MODELS_DIR, CF_ITEM_MAP_FILENAME),
        'item_map_inv': os.path.join(MODELS_DIR, CF_ITEM_MAP_INV_FILENAME),
        'games_df': os.path.join(MODELS_DIR, CF_GAMES_DF_FILENAME),
    }

    all_loaded_successfully = True
    for name, path in required_files.items():
        if not os.path.exists(path):
            st.error(f"Error: CF Artifact file not found at '{path}'.")
            all_loaded_successfully = False
            artifacts_loaded[name] = None
        else:
            try:
                with open(path, 'rb') as f:
                    artifacts_loaded[name] = pickle.load(f)
                print(f"Successfully loaded CF artifact: {name}")
            except Exception as e:
                st.error(f"Error loading CF artifact '{name}' from '{path}': {e}")
                all_loaded_successfully = False
                artifacts_loaded[name] = None

    if not all_loaded_successfully:
        print("One or more CF artifacts failed to load.")
        return None, None, None, None

    print("All CF artifacts loaded successfully.")
    return (
        artifacts_loaded.get('similarity_matrix'),
        artifacts_loaded.get('item_map'),
        artifacts_loaded.get('item_map_inv'),
        artifacts_loaded.get('games_df')
    )


# --- CBF Recommendation Logic ---
def recommend_for_set_content(
    target_app_ids: List[int],
    nn_model: Any, # Type hint as Any, actual type is NearestNeighbors
    tfidf_matrix: scipy.sparse.csr_matrix,
    item_map: Dict[int, int],
    item_map_inv: Dict[int, int],
    games_df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """Recommends games based on content similarity using KNN on TF-IDF vectors."""
    # (Code copied from app_cont.py - unchanged internally)
    if not target_app_ids:
        return pd.DataFrame(columns=['app_id', 'game_name', 'content_similarity'])

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

    profile_vector = tfidf_matrix[input_indices].mean(axis=0)
    if isinstance(profile_vector, np.matrix):
        profile_vector = profile_vector.A
    if np.count_nonzero(profile_vector) == 0:
        print("Warning (CBF): Calculated profile vector is zero.")
        return pd.DataFrame(columns=['app_id', 'game_name', 'content_similarity'])

    n_fetch = top_n + len(input_indices) + 10
    n_fetch = min(n_fetch, tfidf_matrix.shape[0])
    try:
        distances, indices = nn_model.kneighbors(profile_vector.reshape(1, -1), n_neighbors=n_fetch)
    except Exception as e:
        st.error(f"Error during CBF KNN search: {e}")
        return pd.DataFrame(columns=['app_id', 'game_name', 'content_similarity'])

    recommendations = []
    for i, matrix_idx in enumerate(indices[0]):
        rec_app_id = item_map_inv.get(matrix_idx)
        if rec_app_id is not None and rec_app_id not in valid_input_app_ids:
            try:
                # Use the games_df specific to CBF here for lookup
                # Assuming games_df index aligns with matrix_idx after loading
                game_info = games_df.loc[matrix_idx]
                similarity = 1.0 - distances[0][i]
                recommendations.append({
                    'app_id': rec_app_id,
                    'game_name': game_info['title'] if pd.notna(game_info['title']) else "N/A",
                    'content_similarity': float(similarity)
                })
                if len(recommendations) >= top_n:
                    break
            except KeyError:
                 print(f"Warning (CBF): KeyError looking up index {matrix_idx} in games_df.")
            except Exception as e:
                print(f"Warning (CBF): Error processing recommended index {matrix_idx}: {e}")

    return pd.DataFrame(recommendations)


# --- CF Recommendation Logic ---
def recommend_for_set_collaborative(
    target_app_ids: List[int],
    similarity_matrix: np.ndarray,
    item_map: Dict[int, int],
    item_map_inv: Dict[int, int],
    games_df: pd.DataFrame,
    top_n: int = 10 # Default value adjusted for consistency
) -> pd.DataFrame:
    """
    Recommends games based on a list of input game app_ids by aggregating
    their similarity scores using item-based collaborative filtering.
    """
    # (Code copied from app_col.py - renamed function, unchanged internally)
    recommendations = []
    if not target_app_ids:
        return pd.DataFrame(columns=['app_id', 'game_name', 'aggregated_score'])

    n_items = similarity_matrix.shape[0]
    aggregated_scores = np.zeros(n_items, dtype=np.float64)
    input_indices = set()
    valid_input_count = 0

    for app_id in target_app_ids:
        if app_id in item_map:
            item_index = item_map[app_id]
            if 0 <= item_index < n_items:
                aggregated_scores += similarity_matrix[item_index]
                input_indices.add(item_index)
                valid_input_count += 1
            else:
                print(f"Debug Warning (CF): app_id {app_id} mapped to out-of-bounds index {item_index}.")

    if valid_input_count == 0:
        print("Warning (CF): None of the selected app_ids were found in item_map.")
        return pd.DataFrame(columns=['app_id', 'game_name', 'aggregated_score'])

    candidate_items = []
    for item_idx in range(n_items):
        if item_idx in item_map_inv and item_idx not in input_indices:
            candidate_items.append((item_idx, aggregated_scores[item_idx]))

    candidate_items.sort(key=lambda x: x[1], reverse=True)
    top_similar_items = candidate_items[:top_n]

    # Use the games_df specific to CF here for lookup
    # Consider setting index for efficiency if needed: games_df_indexed = games_df.set_index('app_id')
    for matrix_idx, score in top_similar_items:
        try:
            recommended_app_id = item_map_inv[matrix_idx]
            # Use boolean indexing which is safer if index isn't guaranteed
            game_info = games_df[games_df['app_id'] == recommended_app_id]
            if not game_info.empty:
                game_name = game_info['title'].iloc[0]
                recommendations.append({
                    'app_id': recommended_app_id,
                    'game_name': game_name if pd.notna(game_name) else "N/A",
                    'aggregated_score': float(score)
                })
            else:
                print(f"Warning (CF): Metadata not found for recommended app_id {recommended_app_id}")
        except KeyError:
            print(f"Warning (CF): Could not find app_id for matrix index {matrix_idx} in item_map_inv.")
        except IndexError:
             print(f"Warning (CF): IndexError during title lookup for recommended app_id {recommended_app_id}.")

    return pd.DataFrame(recommendations)

# --- Streamlit App Main Function ---
def run_app():
    st.title("🎮 Steam Game Recommender")
    st.markdown("Select a recommendation model and games you like to get suggestions.")

    # --- Model Selection ---
    model_choice = st.radio(
        "Choose Recommendation Model:",
        ("Content-Based Filtering", "Collaborative Filtering"),
        horizontal=True,
        key="model_selector"
    )
    st.divider()

    # --- Load Artifacts Based on Choice ---
    artifacts = None
    games_pd = None
    item_map = None # Need this for populating selector

    if model_choice == "Content-Based Filtering":
        st.subheader("Content-Based Filtering")
        st.markdown("Recommends games similar in **content** (tags, descriptions).")
        artifacts = load_cbf_artifacts()
        # Unpack CBF artifacts carefully checking for None
        if all(a is not None for a in artifacts):
            nn_model, tfidf_matrix, item_map, item_map_inv, games_pd = artifacts
        else:
            st.error("Failed to load one or more CBF artifacts. Cannot proceed.")
            st.stop()

    elif model_choice == "Collaborative Filtering":
        st.subheader("Collaborative Filtering")
        st.markdown("Recommends games based on **user interaction patterns** (what similar users play).")
        artifacts = load_cf_artifacts()
         # Unpack CF artifacts carefully checking for None
        if all(a is not None for a in artifacts):
            item_similarity_matrix, item_map, item_map_inv, games_pd = artifacts
        else:
            st.error("Failed to load one or more CF artifacts. Cannot proceed.")
            st.stop()

    # --- Prepare Data for UI (using the correct games_pd and item_map) ---
    if games_pd is not None and item_map is not None:
        selectable_games_list = []
        # Filter games_pd to only include titles present in the *current model's* item_map
        for app_id, title in games_pd[['app_id', 'title']].values:
            if pd.notna(title) and title.strip() and app_id in item_map:
                 selectable_games_list.append({'app_id': app_id, 'title': title})

        if not selectable_games_list:
            st.error(f"No selectable games found for the {model_choice} model. Check artifact consistency.")
            st.stop()

        selectable_games_df = pd.DataFrame(selectable_games_list).drop_duplicates().sort_values('title')
        game_titles_list = selectable_games_df['title'].tolist()
    else:
        # Should not happen if loading checks passed, but as a safeguard
        st.error("Game data or item map failed to load correctly.")
        st.stop()


    # --- User Input Area ---
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_titles = st.multiselect(
            label=f"Select Game(s) You Like (for {model_choice}):",
            options=game_titles_list,
            placeholder="Type or select game titles...",
            key=f"game_selector_{model_choice.replace(' ', '_')}", # Unique key per model
            label_visibility="collapsed"
        )

    with col2:
        top_n = st.slider(
            "Number of recommendations:",
            min_value=5, max_value=25, value=10, step=1,
            key=f"top_n_slider_{model_choice.replace(' ', '_')}" # Unique key per model
        )
    st.divider()

    # --- Generate and Display Recommendations ---
    if selected_titles:
        try:
            # Get the app_ids for the selected titles using the correct selectable_games_df
            selected_app_ids = selectable_games_df[selectable_games_df['title'].isin(selected_titles)]['app_id'].tolist()

            if selected_app_ids:
                titles_display = ", ".join([f"'{title}'" for title in selected_titles])
                st.subheader(f"Recommendations based on: {titles_display}")

                recommendations_df = pd.DataFrame() # Initialize
                score_col_name = ""
                score_help_text = ""
                score_format = ""

                # --- Call the correct recommendation function ---
                if model_choice == "Content-Based Filtering":
                    nn_model, tfidf_matrix, item_map, item_map_inv, games_pd_cbf = artifacts # Unpack again for clarity
                    recommendations_df = recommend_for_set_content(
                        target_app_ids=selected_app_ids,
                        nn_model=nn_model,
                        tfidf_matrix=tfidf_matrix,
                        item_map=item_map,
                        item_map_inv=item_map_inv,
                        games_df=games_pd_cbf, # Use the correct games_df
                        top_n=top_n
                    )
                    score_col_name = "content_similarity"
                    score_help_text = "Higher score (closer to 1) = More similar content."
                    score_format = "%.4f"


                elif model_choice == "Collaborative Filtering":
                    item_similarity_matrix, item_map, item_map_inv, games_pd_cf = artifacts # Unpack again
                    recommendations_df = recommend_for_set_collaborative(
                        target_app_ids=selected_app_ids,
                        similarity_matrix=item_similarity_matrix,
                        item_map=item_map,
                        item_map_inv=item_map_inv,
                        games_df=games_pd_cf, # Use the correct games_df
                        top_n=top_n
                    )
                    score_col_name = "aggregated_score"
                    score_help_text = "Higher score = More similar based on user interactions."
                    score_format = "%.4f"

                # --- Display Results ---
                if not recommendations_df.empty:
                    # Rename the score column dynamically for display
                    recommendations_df.rename(columns={score_col_name: "Score"}, inplace=True)

                    # --- !!! ADD SEQUENCE NUMBER HERE !!! ---
                    # 1. Create sequence starting from 1
                    recommendations_df.insert(0, 'No.', range(1, len(recommendations_df) + 1))
                    # -----------------------------------------

                    # Define column order explicitly including the new 'No.' column
                    display_columns = ['No.', 'app_id', 'game_name', 'Score']

                    st.dataframe(
                        recommendations_df[display_columns], # Display only the specified columns in order
                        column_config={
                            # --- Add config for 'No.' ---
                            "No.": st.column_config.NumberColumn("No.", help="Recommendation rank", format="%d"),
                            # ----------------------------
                            "app_id": st.column_config.NumberColumn("App ID", format="%d"),
                            "game_name": st.column_config.TextColumn("Recommended Game"),
                            "Score": st.column_config.NumberColumn(
                                "Similarity Score", # Use a generic name now
                                help=score_help_text,
                                format=score_format
                            ),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    st.caption(f"Displaying top {len(recommendations_df)} {model_choice} recommendations.")
                else:
                    st.info(f"No suitable {model_choice} recommendations found for the selected combination. Try different or fewer games.")
            else:
                 st.warning("Could not retrieve App IDs for the selected titles.")

        except Exception as e:
            st.error(f"An unexpected error occurred during {model_choice} recommendation generation.")
            print(f"Error during {model_choice} recommendation generation: {e}")
            import traceback
            print(traceback.format_exc()) # Log detailed traceback to console

    # --- Initial State Message ---
    # Use session state to show message only after first interaction if nothing is selected
    # Key includes model choice to reset message state when switching models
    session_state_key = f"app_ran_once_{model_choice.replace(' ', '_')}"
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = False # Initialize

    if not selected_titles and st.session_state[session_state_key]:
        st.info("Select one or more games above to get recommendations.")

    # Mark that the app has run past the initial setup for the current model
    st.session_state[session_state_key] = True

# --- Run the App ---
if __name__ == "__main__":
    run_app()