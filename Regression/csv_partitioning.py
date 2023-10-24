import pandas as pd

# Reading the first 200 rows from the large CSV file
df = pd.read_csv('dataset/heart_attack_prediction_dataset.csv', nrows=200)

# Saving the 200 rows to another CSV file
df.to_csv('dataset/dataset.csv', index=False)
