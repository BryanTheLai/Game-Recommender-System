import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import os
# os.chdir('C:\\Users\\wbrya\\OneDrive\\Documents\\GitHub\\MovieLens-Recommender-System')
# print("Current working directory:", os.getcwd())

# Read JSON file
json_file_path = 'data/raw/games_metadata.json'
games_metadata_df = pd.read_json(json_file_path, lines=True)

# Read CSV files
games_csv = pd.read_csv('data/raw/games.csv')
recommendations_csv = pd.read_csv('data/raw/recommendations.csv')
users_csv = pd.read_csv('data/raw/users.csv')

print("JSON DataFrame:")
print(games_metadata_df.head())