# Game Recommender System
Put the csv files in data/raw: [Steam Dataset from Kaggle](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)

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
