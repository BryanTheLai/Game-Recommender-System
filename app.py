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

# CF Artifact Filenames (Collaborative Filtering - Original)
CF_SIMILARITY_MATRIX_FILENAME = "item_similarity_matrix.pkl"
CF_ITEM_MAP_FILENAME = "item_map.pkl" # Potentially different item map for CF
CF_ITEM_MAP_INV_FILENAME = "item_map_inv.pkl" # Potentially different inverse map for CF
CF_GAMES_DF_FILENAME = "filtered_games_metadata.pkl" # Potentially different games df for CF

# --- NEW: MF Artifact Filenames (Matrix Factorization - based on the new notebook's output) ---
MF_SIMILARITY_MATRIX_FILENAME = "item_similarity_matrix_cosine.pkl" # From matrix.ipynb output
MF_ITEM_MAP_FILENAME = "item_map_cosine.pkl"                  # From matrix.ipynb output
MF_ITEM_MAP_INV_FILENAME = "item_map_inv_cosine.pkl"              # From matrix.ipynb output
MF_GAMES_DF_FILENAME = "filtered_games_metadata_cosine.pkl"   # From matrix.ipynb output

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
    """Loads Collaborative Filtering artifacts using specific filenames (Original CF)."""
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

# --- NEW: Loading function for MF artifacts ---
@st.cache_data
def load_mf_artifacts() -> Tuple[np.ndarray | None, Dict[int, int] | None, Dict[int, int] | None, pd.DataFrame | None]:
    """Loads 'Matrix Factorization' (Item-Item Cosine Similarity) artifacts using new filenames."""
    print("Attempting to load Matrix Factorization (Cosine Sim) artifacts...")
    artifacts_loaded = {}
    # Use the filenames defined for MF (from the new notebook)
    required_files = {
        'similarity_matrix': os.path.join(MODELS_DIR, MF_SIMILARITY_MATRIX_FILENAME),
        'item_map': os.path.join(MODELS_DIR, MF_ITEM_MAP_FILENAME),
        'item_map_inv': os.path.join(MODELS_DIR, MF_ITEM_MAP_INV_FILENAME),
        'games_df': os.path.join(MODELS_DIR, MF_GAMES_DF_FILENAME),
    }

    all_loaded_successfully = True
    for name, path in required_files.items():
        if not os.path.exists(path):
            st.error(f"Error: Matrix Factorization Artifact file not found at '{path}'.")
            all_loaded_successfully = False
            artifacts_loaded[name] = None
        else:
            try:
                with open(path, 'rb') as f:
                    artifacts_loaded[name] = pickle.load(f)
                print(f"Successfully loaded MF artifact: {name}")
            except Exception as e:
                st.error(f"Error loading MF artifact '{name}' from '{path}': {e}")
                all_loaded_successfully = False
                artifacts_loaded[name] = None

    if not all_loaded_successfully:
        print("One or more MF artifacts failed to load.")
        return None, None, None, None

    print("All MF artifacts loaded successfully.")
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
    # (Code unchanged internally)
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
                # Use boolean indexing for safety if index is not aligned
                game_info_rows = games_df[games_df['app_id'] == rec_app_id]
                if not game_info_rows.empty:
                    game_info = game_info_rows.iloc[0]
                    similarity = 1.0 - distances[0][i]
                    recommendations.append({
                        'app_id': rec_app_id,
                        'game_name': game_info['title'] if pd.notna(game_info['title']) else "N/A",
                        'content_similarity': float(similarity)
                    })
                    if len(recommendations) >= top_n:
                        break
                else:
                    print(f"Warning (CBF): Metadata not found for recommended app_id {rec_app_id}")
            except KeyError:
                 print(f"Warning (CBF): KeyError looking up app_id {rec_app_id} in games_df.")
            except Exception as e:
                print(f"Warning (CBF): Error processing recommended app_id {rec_app_id}: {e}")

    return pd.DataFrame(recommendations)


# --- CF Recommendation Logic (Common function for Similarity Matrix based approaches) ---
def recommend_for_set_similarity_based(
    target_app_ids: List[int],
    similarity_matrix: np.ndarray,
    item_map: Dict[int, int],
    item_map_inv: Dict[int, int],
    games_df: pd.DataFrame,
    top_n: int = 10,
    model_name: str = "Similarity" # For logging/warnings
) -> pd.DataFrame:
    """
    Recommends games based on a list of input game app_ids by aggregating
    their similarity scores using a precomputed item-item similarity matrix.
    (Generic function usable by both original CF and new MF model).
    """
    recommendations = []
    score_col_name = 'aggregated_score' # Standard internal name
    if not target_app_ids:
        return pd.DataFrame(columns=['app_id', 'game_name', score_col_name])

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
                print(f"Debug Warning ({model_name}): app_id {app_id} mapped to out-of-bounds index {item_index}.")
        else:
            print(f"Debug Info ({model_name}): Input app_id {app_id} not found in this model's item_map.")


    if valid_input_count == 0:
        print(f"Warning ({model_name}): None of the selected app_ids were found in item_map or were valid.")
        return pd.DataFrame(columns=['app_id', 'game_name', score_col_name])

    # Normalize scores by the number of valid inputs to get an average similarity? Optional, but might be more intuitive.
    # aggregated_scores /= valid_input_count # Uncomment to average

    candidate_items = []
    for item_idx in range(n_items):
        if item_idx in item_map_inv and item_idx not in input_indices:
            candidate_items.append((item_idx, aggregated_scores[item_idx]))

    candidate_items.sort(key=lambda x: x[1], reverse=True)
    top_similar_items = candidate_items[:top_n]

    # Use the specific games_df passed to this function for lookup
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
                    score_col_name: float(score) # Use the standard internal name
                })
            else:
                print(f"Warning ({model_name}): Metadata not found for recommended app_id {recommended_app_id}")
        except KeyError:
            print(f"Warning ({model_name}): Could not find app_id for matrix index {matrix_idx} in item_map_inv.")
        except IndexError:
             print(f"Warning ({model_name}): IndexError during title lookup for recommended app_id {recommended_app_id}.")
        except Exception as e:
             print(f"Warning ({model_name}): Unexpected error processing recommendation index {matrix_idx}: {e}")


    return pd.DataFrame(recommendations)


# --- Streamlit App Main Function ---
def run_app():
    st.title("🎮 Steam Game Recommender")
    st.markdown("Select a recommendation model and games you like to get suggestions.")

    # --- Model Selection ---
    model_choice = st.radio(
        "Choose Recommendation Model:",
        ( # Add the new option
            "Content-Based Filtering",
            "Collaborative Filtering",
            "Matrix Factorization" # Display name for the new model
        ),
        horizontal=True,
        key="model_selector"
    )
    st.divider()

    # --- Load Artifacts Based on Choice ---
    artifacts = None
    games_pd = None
    item_map = None # Need this for populating selector
    model_description = ""
    model_load_func = None
    recommendation_func = None
    specific_args = {} # To hold model-specific artifacts needed by the rec function

    if model_choice == "Content-Based Filtering":
        st.subheader("Content-Based Filtering")
        model_description = "Recommends games similar in **content** (tags, descriptions)."
        model_load_func = load_cbf_artifacts
        recommendation_func = recommend_for_set_content # CBF uses its own function
        artifacts = model_load_func()
        if all(a is not None for a in artifacts):
            nn_model, tfidf_matrix, item_map, item_map_inv, games_pd = artifacts
            # Store specific artifacts needed by recommend_for_set_content
            specific_args = {
                'nn_model': nn_model,
                'tfidf_matrix': tfidf_matrix,
                'item_map': item_map,
                'item_map_inv': item_map_inv,
                'games_df': games_pd # Pass the correct games_df
            }
        else:
            st.error("Failed to load one or more CBF artifacts. Cannot proceed.")
            st.stop()

    elif model_choice == "Collaborative Filtering":
        st.subheader("Collaborative Filtering (Original)")
        model_description = "Recommends games based on **user interaction patterns** (what similar users play - original model)."
        model_load_func = load_cf_artifacts
        recommendation_func = recommend_for_set_similarity_based # Use generic similarity func
        artifacts = model_load_func()
        if all(a is not None for a in artifacts):
            item_similarity_matrix, item_map, item_map_inv, games_pd = artifacts
            # Store specific artifacts needed by recommend_for_set_similarity_based
            specific_args = {
                'similarity_matrix': item_similarity_matrix,
                'item_map': item_map,
                'item_map_inv': item_map_inv,
                'games_df': games_pd, # Pass the correct games_df
                'model_name': "CF (Original)"
            }
        else:
            st.error("Failed to load one or more CF artifacts. Cannot proceed.")
            st.stop()

    # --- NEW: Handle Matrix Factorization Choice ---
    elif model_choice == "Matrix Factorization":
        st.subheader("Matrix Factorization (Interaction Similarity)")
        model_description = "Recommends games based on **user interaction patterns** (derived from interaction matrix similarity - new model)."
        st.caption("_Note: This implementation uses Item-Item Cosine Similarity derived from user interaction data, similar to Collaborative Filtering but potentially trained on different data/parameters._")
        model_load_func = load_mf_artifacts # Use the new loading function
        recommendation_func = recommend_for_set_similarity_based # Use generic similarity func
        artifacts = model_load_func()
        if all(a is not None for a in artifacts):
            item_similarity_matrix_mf, item_map_mf, item_map_inv_mf, games_pd_mf = artifacts
            # Store specific artifacts needed by recommend_for_set_similarity_based
            specific_args = {
                'similarity_matrix': item_similarity_matrix_mf,
                'item_map': item_map_mf,
                'item_map_inv': item_map_inv_mf,
                'games_df': games_pd_mf, # Pass the correct games_df for MF
                'model_name': "MF (Cosine Sim)"
            }
            # Update the main variables used for UI population
            item_map = item_map_mf
            games_pd = games_pd_mf
        else:
            st.error("Failed to load one or more Matrix Factorization artifacts. Cannot proceed.")
            st.stop()
    # --- END NEW SECTION ---

    st.markdown(model_description) # Display the description for the selected model

    # --- Prepare Data for UI (using the correct games_pd and item_map for the selected model) ---
    if games_pd is not None and item_map is not None:
        selectable_games_list = []
        # Filter games_pd to only include titles present in the *current model's* item_map
        # Ensure we handle missing titles gracefully
        for app_id, title in games_pd[['app_id', 'title']].values:
            # Check title is not null/empty and app_id is in the current model's map
            if pd.notna(title) and str(title).strip() and app_id in item_map:
                 selectable_games_list.append({'app_id': app_id, 'title': str(title).strip()})

        if not selectable_games_list:
            st.error(f"No selectable games found for the {model_choice} model. Check artifact consistency (e.g., ensure '{specific_args.get('games_df_path', 'games_df')}' matches the item map).")
            st.stop()

        selectable_games_df = pd.DataFrame(selectable_games_list).drop_duplicates(subset=['app_id']).sort_values('title') # Ensure unique app_ids
        game_titles_list = selectable_games_df['title'].tolist()
    else:
        # Should not happen if loading checks passed, but as a safeguard
        st.error("Game data or item map failed to load correctly after model selection.")
        st.stop()


    # --- User Input Area ---
    col1, col2 = st.columns([3, 1])

    with col1:
        # Use a unique key based on model_choice to reset selection when model changes
        select_key = f"game_selector_{model_choice.replace(' ', '_').replace('(', '').replace(')', '')}"
        selected_titles = st.multiselect(
            label=f"Select Game(s) You Like (for {model_choice}):",
            options=game_titles_list,
            placeholder="Type or select game titles...",
            key=select_key,
            label_visibility="collapsed"
        )

    with col2:
        # Use a unique key based on model_choice
        slider_key = f"top_n_slider_{model_choice.replace(' ', '_').replace('(', '').replace(')', '')}"
        top_n = st.slider(
            "Number of recommendations:",
            min_value=5, max_value=25, value=10, step=1,
            key=slider_key
        )
    st.divider()

    # --- Generate and Display Recommendations ---
    if selected_titles:
        try:
            # Get the app_ids for the selected titles using the correct selectable_games_df for the current model
            selected_app_ids = selectable_games_df[selectable_games_df['title'].isin(selected_titles)]['app_id'].tolist()

            if selected_app_ids:
                titles_display = ", ".join([f"'{title}'" for title in selected_titles])
                st.subheader(f"Recommendations based on: {titles_display}")

                recommendations_df = pd.DataFrame() # Initialize
                score_col_name = "Score" # Default display name
                score_help_text = ""
                score_format = ""
                internal_score_col = "" # The actual column name returned by the function

                # --- Call the correct recommendation function with its specific args ---
                if recommendation_func:
                    recommendations_df = recommendation_func(
                        target_app_ids=selected_app_ids,
                        top_n=top_n,
                        **specific_args # Unpack the specific artifacts for the function
                    )
                else:
                     st.error("Internal Error: Recommendation function not set.") # Should not happen

                # --- Set score column details based on model ---
                if model_choice == "Content-Based Filtering":
                    internal_score_col = "content_similarity"
                    score_col_name = "Content Score"
                    score_help_text = "Higher score (closer to 1) = More similar content (TF-IDF Cosine Sim)."
                    score_format = "%.4f"
                elif model_choice == "Collaborative Filtering":
                    internal_score_col = "aggregated_score"
                    score_col_name = "Interaction Score"
                    score_help_text = "Higher score = Stronger predicted affinity based on user interactions (Original CF)."
                    score_format = "%.4f"
                elif model_choice == "Matrix Factorization":
                    internal_score_col = "aggregated_score"
                    score_col_name = "Interaction Score"
                    score_help_text = "Higher score = Stronger predicted affinity based on user interactions (MF Model - Cosine Sim)."
                    score_format = "%.4f"


                # --- Display Results ---
                if not recommendations_df.empty and internal_score_col in recommendations_df.columns:
                    # Rename the score column dynamically for display
                    recommendations_df.rename(columns={internal_score_col: score_col_name}, inplace=True)

                    # Create sequence number starting from 1
                    recommendations_df.insert(0, 'No.', range(1, len(recommendations_df) + 1))

                    # Define column order explicitly including the new 'No.' column and renamed score column
                    display_columns = ['No.', 'app_id', 'game_name', score_col_name]

                    st.dataframe(
                        recommendations_df[display_columns], # Display only the specified columns in order
                        column_config={
                            "No.": st.column_config.NumberColumn("No.", help="Recommendation rank", format="%d"),
                            "app_id": st.column_config.NumberColumn("App ID", format="%d"),
                            "game_name": st.column_config.TextColumn("Recommended Game"),
                            score_col_name: st.column_config.NumberColumn( # Use dynamic name here
                                score_col_name, # Use the dynamic name as the label too
                                help=score_help_text,
                                format=score_format
                            ),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    st.caption(f"Displaying top {len(recommendations_df)} {model_choice} recommendations.")
                elif not recommendations_df.empty:
                     st.error(f"Internal Error: Expected score column '{internal_score_col}' not found in results.")
                     print(f"Debug: Columns found: {recommendations_df.columns}") # Debug print
                else:
                    st.info(f"No suitable {model_choice} recommendations found for the selected combination. Try different or fewer games.")
            else:
                 st.warning("Could not retrieve App IDs for the selected titles. Check if titles exist in the selected model's data.")

        except Exception as e:
            st.error(f"An unexpected error occurred during {model_choice} recommendation generation.")
            print(f"Error during {model_choice} recommendation generation: {e}")
            import traceback
            print(traceback.format_exc()) # Log detailed traceback to console

    # --- Initial State Message ---
    # Use session state to show message only after first interaction if nothing is selected
    session_state_key = f"app_ran_once_{model_choice.replace(' ', '_').replace('(', '').replace(')', '')}"
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = False # Initialize

    if not selected_titles and st.session_state[session_state_key]:
        st.info("Select one or more games above to get recommendations.")

    # Mark that the app has run past the initial setup for the current model
    st.session_state[session_state_key] = True

# --- Run the App ---
if __name__ == "__main__":
    # Create models directory if it doesn't exist (optional, good practice)
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        st.warning(f"Created models directory at '{MODELS_DIR}'. Ensure your model artifact files (.pkl, .npy) are placed here.")

    run_app()