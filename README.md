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

## Metrics
* Precision
* Recall
* F1 Score


## Project Structure

```
├── data/
│   ├── external/    # API data
│   ├── interim/     # Transformed data
│   ├── processed/   # Final data for modeling
│   └── raw/         # Original data
├── docs/
│   └── README.md    # Project documentation
├── models/
│   └── *.pkl       # Trained models (e.g., pickled)
├── notebooks/
│   └── [NN]-*.ipynb # Jupyter notebooks (e.g., 01-data-exploration.ipynb)
├── references/
│   └── *.pdf        # Data dictionaries, manuals
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

| Concept                 | Definition in Recommender Context                                                                 | Relation to Code / Formula                                                                   |
| :---------------------- | :------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------- |
| **K**                   | The **number** of top recommendations you are evaluating (e.g., K=10, K=20).                      | Parameter `k` in `precision_at_k`, `recall_at_k`, and `recommend_for_user` functions.      |
| **Relevant Items**      | The items the user *actually interacted with* in the **test set** (hidden data).                  | `actual_items` (a `Set[int]` for a specific user from `filtered_test_items_map`).            |
| **Recommended Items @K** | The **top K** items suggested by your recommendation function for a specific user.                 | `recommended_items[:k]` (a `List[int]`, the first `k` items from `recommend_for_user`).       |
| **True Positives (TP)** | Items **correctly recommended**: they are in the top K recommendations AND are relevant (in test set). | `hits = len(set(recommended_items[:k]) & actual_items)`                                       |
| **False Positives (FP)**| Items **incorrectly recommended**: they are in the top K recommendations BUT are NOT relevant.      | `k - hits` (Items in the top K list that are not in `actual_items`).                         |
| **False Negatives (FN)**| Items **missed**: they are relevant (in test set) BUT are NOT in the top K recommendations.        | `len(actual_items) - hits` (Relevant items that were not among the top K recommendations). |
|                         |                                                                                                   |                                                                                              |
| **Precision@K**         | **Accuracy of recommendations:** Of the K items recommended, what fraction were actually relevant?    | `hits / k` <br> (Equivalent to `TP / (TP + FP)`)                                               |
| **Recall@K**            | **Completeness of recommendations:** Of all the relevant items, what fraction did you recommend in the top K? | `hits / len(actual_items)` <br> (Equivalent to `TP / (TP + FN)`)                               |



```
user_median_interactions = int(user_interaction_counts.median())
item_median_interactions = int(item_interaction_counts.median())
print(f"User Median: {user_median_interactions}")
print(f"Item Median: {user_median_interactions}")
```