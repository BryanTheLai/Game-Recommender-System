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
