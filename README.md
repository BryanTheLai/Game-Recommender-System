# Steps to run

1. Create and activate a virtual environment  
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
2. Install project dependencies  
    ```bash
    pip install -r requirements.txt
    ```
3. Launch the Streamlit app  
    ```bash
    streamlit run app.py
    ```  

# Game Recommender System
Put the csv files in data/raw: [Steam Dataset from Kaggle](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)

The Steam Store dataset contains over 41 million cleaned and preprocessed user recommendations (reviews).
**Steam Store** is a leading online platform for purchasing and downloading gaming-related content. 
Contains detailed information about games and add-ons.

## Models
Top N Game Recommendations and using three models 
1. Collaborative Filtering
2. Content-Based			 
3. Matrix Factorization


    ## Project Structure ([Cookie Cuter Template](https://github.com/drivendataorg/cookiecutter-data-science))

    ```
    ├── data/
    │   ├── external/    # API data
    │   ├── interim/     # Transformed data
    │   ├── processed/   # Final data for modeling
    │   └── raw/         # Original data
    ├── models/
    │   └── *.pkl       # Trained models (e.g., pickled)
    ├── notebooks/
    │   └── [NN]-*.ipynb # Jupyter notebooks (e.g., 01-data-exploration.ipynb)
    └── requirements.txt # Project dependencies
    ```

    ## Data Exploration
    ```
    games.csv:
    app_id			-Int		- product id of game 
    title			-String		- the name of game
    date_release		-Date		- the date that the game release
    win			-Boolean	- the game supported on Windows 
    mac			-Boolean	- the game supported on MacOS
    Linux			-Boolean	- the game supported on Linux
    rating			-String		- the rating given by users
    positive ratio		-int (%)	- the percentage rating by users
    user_review		-int		- total users that review the game
    price_final		-float		- Price in US dollars $ calculated after the discount
    price_original		-float		- original price before discount(if any)
    discount		-int(%)		- discount on that game(if any)
    steam_deck		-Boolean	- the game supported on steam deck



    users.csv:
    user_id			-Int		-user id
    products		-Int		-no. of game purchased by the user
    reviews			-Int		-no. of reviews 


    recommendations.csv:
    app_id			-Int 		- product id of the game
    helpful			-Int		- no. of users found that a recommendation helpful
    funny			-Int		- no. of users found that a recommendation funny
    date			-Date		- date of publishing
    is_recommended		-Boolean	- whether the user recommend the product
    hours			-time		- hours playing on that game
    user_id			-Int		- user id
    review_id		-Int		- review id (auto-generated)


    games_metadata.json:
    app_id			-Int 		- product id of the game
    description		-String		- brief description about the game 
    tags			-String		- categories/style of the game
    ```


| Concept                 | Definition in Recommender Context                                                                            | Relation to Code / Formula                                                                        |
| :---------------------- | :----------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ |
| **Interaction**         | The specific user action defining a positive relationship (e.g., wrote a review, `is_recommended=True`, played > X hours). | The entries forming the user-item matrix.                                                         |
| **K**                   | The **number** of top recommendations generated and evaluated (e.g., K=20).                                     | Parameter `K` in evaluation functions.                                                            |
| **Relevant Items**      | The items the user *actually interacted with* (as per Interaction definition) in the **test set**.                | `actual_items` (Set of item IDs for a user from the test split).                                  |
| **Recommended Items @K**| The **top K** items suggested by the model for a user, based on training data.                              | `recommended_ids[:K]` (List of item IDs output by `recommend_for_user`).                            |
| **True Positives (TP) @K** | Items **correctly recommended**: present in both the Top K Recommended list AND the Relevant Items list.        | `hits = len(set(recommended_ids[:K]) & actual_items)`                                            |
| **Precision@K**         | **Accuracy of recommendations:** What *fraction* of the K recommended items were actually relevant (TPs)?         | `hits / K` <br> (How relevant is the list shown?)                                                |
| **Recall@K**            | **Coverage of recommendations:** What *fraction* of all the user's Relevant Items were captured in the top K list? | `hits / len(actual_items)` <br> (How much of the user's interest did we find?)                  |


**Example:**

Let's say we evaluate for **User A** with **K=5**.

1.  **Relevant Items (from Test Set):** User A actually interacted with `{Game_A, Game_B, Game_C, Game_D}` in the test data. So, `len(actual_items) = 4`.
2.  **Recommended Items @K=5:** Your model recommends the following top 5 games: `{Game_A, Game_X, Game_C, Game_Y, Game_Z}`.
3.  **Calculating Hits (True Positives):** We compare the two lists:
    *   `Game_A` is in both lists (TP).
    *   `Game_C` is in both lists (TP).
    *   `Game_X`, `Game_Y`, `Game_Z` were recommended but are *not* in the relevant list (False Positives, FP).
    *   `Game_B`, `Game_D` are relevant but were *not* recommended in the top 5 (False Negatives, FN).
    *   The number of **True Positives (hits)** is **2**.
4.  **Calculating Metrics:**
    *   **Precision@5** = Hits / K = 2 / 5 = **0.40** (40% of the recommendations were correct).
    *   **Recall@5** = Hits / Total Relevant Items = 2 / 4 = **0.50** (50% of the relevant items were captured in the top 5 recommendations).

---
