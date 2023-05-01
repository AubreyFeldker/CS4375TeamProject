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

df = df.sort_values(by = ['Latitude', 'Longitude', 'Timestamp'])
df = df.reset_index()

print(df[['Latitude', 'Longitude']].value_counts())

split_ratio = 0.8
split_index = int(len(df) * split_ratio)

df_groups = df.groupby(['Latitude', 'Longitude'])

rnn = RNN(internal_layers=3, input_nodes=len(df.columns) - 1, output_nodes=7)

seq_length = 10

# Applying the neural network across each set of lat/long, since they are constants during a recursive network
for name, group in df_groups:
    # 80/20 test-train split along each group
    split_index = int(len(group) * split_ratio)
    #rnn.train(training_data=group[:split_index], epochs=4, sequence_length=10)
    #group = group.drop(['Timestamp'], axis = 1)

    # Normalization of timestamps to minutes
    group['Timestamp'] = group['Timestamp'] / 60000

    

    rnn.train(train_items=group[:split_index], sequence_length=seq_length)
    break
    
for name, group in df_groups:
    split_index = int(len(group) * split_ratio)
    group['Timestamp'] = group['Timestamp'] / 60000

    rnn.test(test_items=group[split_index:], sequence_length=seq_length)
    