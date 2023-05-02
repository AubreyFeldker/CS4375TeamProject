# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import math

# Preprocessing
import time
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from RecurrentNeuralNetwork import RNN

df = pd.read_csv('https://raw.githubusercontent.com/AubreyFeldker/CS4375TeamProject/main/processed_weather_df.csv')

# Removes rows missing values in the one-hot
df[df['Cold'].astype(bool)]  

n_remove = 30000
df = df.sample(n=len(df)-n_remove).sort_index()
df = df.sort_values(by = ['Latitude', 'Longitude', 'Timestamp'])
df = df.reset_index().drop(['index'], axis=1)

split_ratio = 0.8
split_index = int(len(df) * split_ratio)

df_groups = df.groupby(['Latitude', 'Longitude'])

rnn = RNN(internal_layers=3, input_nodes=len(df.columns) - 1, output_nodes=7)

training_seq_length = 10

internal_layers_list = [1, 3]
update_weight_list = [.0005, 0.0001, .00005]
sequence_length_list = [10, 15, 20]
results = {}

for internal_layers in internal_layers_list:
    for update_weight in update_weight_list:
        
        print(f"Testing internal_layers: {internal_layers}, update_weight: {update_weight}")

        rnn = RNN(internal_layers=internal_layers, input_nodes=len(df.columns) - 1, output_nodes=7, update_weight=update_weight)

        # Training the RNN
        train_r2_scores = []
        train_mse_scores = []
        for name, group in df_groups:
            # 80/20 test-train split along each group
            split_index = int(len(group) * split_ratio)
            # Normalization of timestamps to minutes
            group['Timestamp'] = group['Timestamp'] / 60000
            r2, mse = rnn.train(train_items=group[:split_index], sequence_length=training_seq_length)
            train_r2_scores.append(r2)
            train_mse_scores.append(mse)


        for seq_length in sequence_length_list:

            # Testing the RNN
            test_r2_scores = []
            test_mse_scores = []
            for name, group in df_groups:
                split_index = int(len(group) * split_ratio)
                group['Timestamp'] = group['Timestamp'] / 60000
                r2, mse = rnn.test(test_items=group[split_index:], sequence_length=seq_length)
                test_r2_scores.append(r2)
                test_mse_scores.append(mse)

            # Storing the results
            key = (internal_layers, update_weight, seq_length)
            results[key] = {
                "train_mse": np.mean(train_mse_scores),
                "test_mse": np.mean(test_mse_scores)
            }


            print(results)
    