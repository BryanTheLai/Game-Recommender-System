import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import os
# os.chdir('C:\\Users\\wbrya\\OneDrive\\Documents\\GitHub\\MovieLens-Recommender-System')
# print("Current working directory:", os.getcwd())

def explore_csv(path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path, lines=True)

    column = df.dtypes.reset_index()
    column.columns = ['Column Name', 'Data Type']

    num_rows = df.shape[0]

    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ['Column Name', 'Missing Values']

    result = column.merge(missing_values, on='Column Name')

    print(f"File: {file_path}")
    print(f"Total Rows: {num_rows}\n")
    print(result)

file_path = "data/raw/games.csv"
explore_csv(file_path)

file_path = "data/raw/recommendations.csv"
explore_csv(file_path)

file_path = "data/raw/users.csv"
explore_csv(file_path)

file_path = "data/raw/games_metadata.json"
explore_csv(file_path)
