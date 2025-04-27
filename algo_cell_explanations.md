Break down the three Jupyter notebooks (`collaborative_filtering.ipynb`, `content-based.ipynb`, `matrix.ipynb`) step-by-step, grouping common operations and highlighting the unique logic for each approach.

**Overall Goal:** These notebooks aim to build and evaluate different types of game recommendation systems using Steam data. They explore:
1.  **Collaborative Filtering (Playtime Weighted):** Recommending games based on similarities in user *engagement levels* (scaled playtime) across games.
2.  **Content-Based Filtering:** Recommending games based on similarities in their *metadata* (title, tags).
3.  **Matrix Factorization (Implemented as Binary Interaction CF):** Recommending games based on similarities in *who positively recommended* the games (binary interaction).

---

**Part 1: Common Steps Across All Notebooks**

These steps appear with nearly identical logic and purpose in all three notebooks, establishing the foundation for each specific model.

*   **Cell 1: Setup and Initial Data Loading**
    *   **What:** Imports necessary libraries (pandas, numpy, sklearn components, os, gc, pickle, etc.) and loads the primary datasets: `recommendations.csv` (user-item interactions) and `games_tagged.csv` (game metadata).
    *   **How:** Uses `pd.read_csv()`. Sets base directory paths. Explicitly casts `app_id` to integer.
    *   **Why:** To bring the raw data into memory and prepare the environment.
    *   **Key Filtering:** Immediately filters the `recommendations` DataFrame to keep **only positive interactions** (`is_recommended_binary == 1`). This focuses all subsequent analysis and modeling on explicit signals of user preference.
    *   **Content-Based Difference:** `content-based.ipynb` additionally uses `ast.literal_eval` to parse the string representation of game `tags` into Python lists, essential for using tags as features later. It also creates a combined `tags_str` column.

*   **Cell 2 & 2b: Interaction Analysis and Visualization**
    *   **What:** Calculates the number of positive interactions (`is_recommended_binary == 1`) per unique `user_id` and `app_id`. Displays descriptive statistics and plots the distribution of these counts on a log scale. Calculates interaction count percentiles.
    *   **How:** Uses `value_counts()`, `.describe()`, `.quantile()`, and a custom `plot_interaction_distribution` function with `matplotlib`.
    *   **Why:** To understand the data's characteristics, specifically the distribution of user activity and item popularity (based *only* on positive recommendations). The plots (Figures 3.3.1 & 3.3.2 mentioned in the report) reveal a typical "long tail" distribution – many users/items have very few positive interactions, while a few have many. This highlights data sparsity. Percentiles help inform filtering decisions.

*   **Cell 3: Filtering by Interaction Thresholds**
    *   **What:** Filters the interaction data further, removing users and items that fall below minimum positive interaction counts.
    *   **How:** Defines thresholds `MIN_USER_INTERACTIONS = 9` and `MIN_ITEM_INTERACTIONS = 2476`. Identifies users and items meeting these thresholds using the counts from Cell 2. Filters the interaction DataFrame using `.isin()`.
    *   **Why & Justification:** This is a crucial step to mitigate data sparsity and improve model reliability.
        *   **Why filter?** Collaborative models struggle with users/items having very few interactions (the "cold start" problem within the existing data). Predictions based on minimal data are noisy and unreliable. Filtering focuses the model on entities where more robust patterns might exist.
        *   **Why these values (9, 2476)?** These thresholds were chosen based on the percentile analysis in Cell 2b. They correspond approximately to the **95th percentile** of user and item positive interaction counts, respectively. This means we are keeping the top 5% most active users and top 5% most recommended items (in terms of positive recommendations). This is a common heuristic approach – aiming to remove the extreme "long tail" while retaining a significant portion of the data (~9.3 million interactions remain) for modeling. It's a trade-off between data volume and data quality/density.

*   **Cell 4: Shuffling Data**
    *   **What:** Randomly shuffles the rows of the filtered interaction DataFrame.
    *   **How:** Uses `.sample(frac=1, random_state=SEED)`. `SEED=42` ensures reproducibility.
    *   **Why:** To ensure that any subsequent operations (like train-test split) are performed on randomized data, preventing potential biases related to the original data ordering.

*   **Cell 5: Finalizing Datasets**
    *   **What:** Gets the unique game `app_id`s present in the final, filtered, shuffled interaction data. Filters the game *metadata* DataFrame (`games_pd` or `games`) to keep only metadata for these specific games (1,872 unique games). Displays sample data.
    *   **How:** Extracts unique IDs using `.unique()`. Filters metadata using `.isin()`.
    *   **Why:** To ensure that the metadata used (especially for looking up titles or, in CBF, for building features) corresponds exactly to the items included in the interaction data used for modeling. This prevents errors and keeps data consistent.

*   **Cell 9 (Conceptually): Saving Artifacts**
    *   **What:** Saves the essential components (trained models, matrices, mappings, metadata) needed to use the recommender later (e.g., in the Streamlit app).
    *   **How:** Uses the `pickle` library to serialize Python objects to files (`.pkl`). Creates a `models` directory if needed.
    *   **Why:** To persist the results of the time-consuming training/calculation steps, allowing the recommendation logic to be deployed or reused without rerunning the entire notebook.
    *   **Difference:** The *specific files* saved differ based on what each model requires (e.g., `item_similarity_matrix` for CF/MF vs. `content_nn_model` and `tfidf_matrix` for CBF). Filenames are often adjusted slightly (e.g., `_cosine` suffix in the MF notebook) to distinguish artifacts.

*   **Cell 10: Train-Test Split for Evaluation**
    *   **What:** Splits the final interaction data into training (80%) and testing (20%) sets for evaluating model performance.
    *   **How:** Uses `groupby('user_id').sample(frac=test_fraction)` to ensure that the split is done *per user* (i.e., 20% of *each* user's interactions go to the test set, if possible). Creates dictionaries (`train_items_map`, `test_items_map`) mapping users to the set of item IDs they interacted with in each split. Filters the test users (`valid_test_user_ids`) to include only those present in both train and test sets *and* having a minimum number of interactions in the training set (ensuring the model has *something* to learn from before testing).
    *   **Why:** Essential for offline evaluation. We train the model on the training set and test its ability to predict the interactions in the (unseen during training) test set for the same users. Splitting by user simulates predicting future interactions for known users. Filtering valid test users ensures the evaluation is meaningful.

*   **Cell 12: Evaluation Metric Functions**
    *   **What:** Defines standard functions to calculate Precision@K and Recall@K.
    *   **How:** Implements the mathematical formulas for Precision (hits / K) and Recall (hits / total relevant items in test set) for the top K recommendations.
    *   **Why:** To provide quantitative measures of recommendation accuracy. `K=20` is chosen as the number of recommendations to evaluate.

*   **Cell 13 / 14: Running Evaluation**
    *   **What:** Executes the evaluation loop.
    *   **How:** Iterates through the `valid_test_user_ids`. For each user:
        1.  Retrieves their actual interactions from the `filtered_test_items_map`.
        2.  Calls the appropriate `recommend_for_user...` function (specific to the model) to generate Top-K recommendations based *only* on the user's training data (`train_items_map`).
        3.  Calculates Precision@K and Recall@K by comparing recommendations to the actual test items.
        4.  Stores the metrics for each user.
        After the loop, calculates and prints the average Precision@K and Recall@K across all evaluated users.
    *   **Why:** To systematically assess the performance of the specific recommendation algorithm on the held-out test data according to the chosen metrics.
    *   **CBF Difference:** Due to the potential computational cost of its vectorized evaluation function, the CBF evaluation loop runs only on a *sample* (`EVALUATION_SAMPLE_SIZE = 10000`) of the valid test users.

---

**Part 2: Model-Specific Logic**

These are the core cells that differentiate the notebooks.

**2.1 `collaborative_filtering.ipynb` (Item-Item CF, Playtime Weighted)**

*   **Cell 6: User-Item Interaction Matrix (Weighted)**
    *   **What:** Constructs the primary data structure for CF.
    *   **How:** Uses `pivot_table` to create a matrix where rows are `user_id`, columns are `app_id`. Crucially, the *value* in the matrix cell `(u, i)` is the `hours_log_scaled` for the positive interaction between user `u` and item `i`. Missing/non-positive interactions are filled with `0`. Converts to CSR sparse matrix. Creates `item_map` (app_id -> matrix col index) and `item_map_inv`.
    *   **Why:** To represent user preferences quantitatively, where higher `hours_log_scaled` indicates stronger engagement for a recommended game. Sparsity requires CSR format. Maps are needed for lookup.

*   **Cell 7: Item-Item Cosine Similarity (Weighted)**
    *   **What:** Calculates similarity between games based on shared user engagement patterns.
    *   **How:** Transposes the user-item matrix (`.T`) so rows represent items. Applies `cosine_similarity` to these item rows (which now contain weighted user interactions). Stores the result in a dense `item_similarity_matrix`.
    *   **Why:** This is the core of Item-Item CF. Cosine similarity on these vectors measures how similar the *weighted* interaction patterns are between pairs of games. Games interacted with similarly (by the same users, with similar engagement levels) get high scores.

*   **Cell 8: Recommendation Function (Weighted)**
    *   **What:** Defines how to get recommendations for a single game.
    *   **How:** The `recommend_similar_games_cosine` function looks up the target game's row in the pre-computed `item_similarity_matrix`, sorts the similarity scores, excludes the game itself, and returns the Top-N most similar games.
    *   **Why:** To operationalize the model for generating "More Like This" recommendations efficiently using the stored similarities.

*   **Cell 11: User Recommendation Function (Evaluation - Weighted)**
    *   **What:** Defines how to get recommendations for a *user* during evaluation.
    *   **How:** The `recommend_for_user` function takes a `user_id`, finds their *training* items, and sums the similarity vectors (rows from `item_similarity_matrix`) corresponding to those training items. This aggregates evidence from items the user liked. It then ranks *other* items based on these aggregated scores and returns the Top-K, excluding training items.
    *   **Why:** To simulate predicting new items for a user based on their known positive interactions and the learned item similarities.

**2.2 `content-based.ipynb` (TF-IDF + KNN)**

*   **Cell 6: Content Feature Preparation (TF-IDF)**
    *   **What:** Converts game titles and tags into numerical feature vectors.
    *   **How:** Concatenates `title` and `tags_str` into a `content` string for each game. Uses `TfidfVectorizer` (with stop words, ngrams, min_df) to transform this text data into a sparse `tfidf_matrix`. Each row represents a game, each column represents a term/ngram weighted by TF-IDF. Creates `item_map`, `item_map_inv` based on the *metadata DataFrame index*.
    *   **Why:** To represent game content numerically, enabling similarity calculations based on shared vocabulary and tag patterns. TF-IDF weights terms by importance.

*   **Cell 7: Fit Nearest Neighbors Model (KNN)**
    *   **What:** Creates an efficient index for finding similar content vectors.
    *   **How:** Fits a `NearestNeighbors` model (using `cosine` metric) on the `tfidf_matrix`.
    *   **Why:** Provides a fast way to query for the "nearest" (most similar) game vectors in the high-dimensional TF-IDF space without pre-calculating all pairwise similarities.

*   **Cell 8: Recommendation Function (Content)**
    *   **What:** Defines how to get recommendations for a single game based on content.
    *   **How:** The `recommend_similar_games_content` function finds the target game's TF-IDF vector index. It uses the fitted `nn_model.kneighbors()` method to find the indices and distances of the most similar vectors (games) directly from the TF-IDF matrix. Returns Top-N based on lowest distance (highest similarity).
    *   **Why:** To generate "More Like This" recommendations based purely on content similarity using the efficient KNN search.

*   **Cell 11: User Recommendation Function (Evaluation - Content)**
    *   **What:** Defines how to get recommendations for a *user* based on content during evaluation.
    *   **How:** The `recommend_for_user_content_vectorized` function calculates an *average* TF-IDF vector representing the user's profile based on their *training* items. It then performs a *single* KNN search using this profile vector against all game vectors in the `tfidf_matrix`. Filters out training items and returns the Top-K nearest neighbors.
    *   **Why:** To evaluate how well content features predict user preference by finding items whose content matches the aggregated content profile of games the user liked previously. Vectorization makes the evaluation loop faster.

**2.3 `matrix.ipynb` (Item-Item CF, Binary Interaction)**

*Note: This follows the CF structure closely, differing mainly in the *values* used in the interaction matrix.*

*   **Cell 7: User-Item Interaction Matrix (Binary)**
    *   **What:** Constructs the user-item matrix using binary interaction signals.
    *   **How:** Uses `pivot_table` similar to the first CF notebook, but the *value* is simply `is_recommended_binary` (which is always `1` after initial filtering) or just a constant `1`. Missing/non-positive interactions are `0`. Converts to CSR sparse matrix. Creates maps.
    *   **Why:** To represent interactions based purely on the *presence* of a positive recommendation, ignoring engagement level (playtime).

*   **Cell 8: Item-Item Cosine Similarity (Binary)**
    *   **What:** Calculates similarity between games based *only* on which users co-recommended them positively.
    *   **How:** Transposes the binary user-item matrix. Applies `cosine_similarity`. Stores result in `item_similarity_matrix`.
    *   **Why:** Similarity score now reflects the overlap in the sets of users who positively recommended pairs of games, treating each positive recommendation equally.

*   **Cell 9: Recommendation Function (Binary)**
    *   **What:** Defines single-game recommendations.
    *   **How:** Identical logic to `recommend_similar_games_cosine` (Cell 8 in first CF notebook), but operates on the similarity matrix derived from *binary* interactions.
    *   **Why:** Generates "More Like This" based on co-recommendation patterns.

*   **Cell 12: User Recommendation Function (Evaluation - Binary)**
    *   **What:** Defines user recommendations for evaluation.
    *   **How:** Identical logic to `recommend_for_user_evaluation` (Cell 11/12 in first CF notebook), aggregating scores from the *binary interaction* similarity matrix based on user's training items.
    *   **Why:** To evaluate the predictive power of the binary co-recommendation patterns.

*   **Cell 15: Generate Top-N Similar Games CSV**
    *   **What:** Pre-computes and saves the Top-10 most similar games for every game in the model.
    *   **How:** Iterates through all items, calls `recommend_similar_games_cosine` for each, stores results, and saves to `top10_similar_games_cosine.csv`.
    *   **Why:** Creates a static lookup table for "More Like This" functionality, useful if runtime calculation is undesirable. This step was specific to this notebook's structure.

---

In essence, all three notebooks follow a similar high-level process: Load -> Filter -> Analyze -> Build Model Representation -> Evaluate -> Save. The core differences lie in *how* item similarity or user preference is represented and calculated: weighted interactions (CF), content features (CBF), or binary interactions (MF notebook's implementation).