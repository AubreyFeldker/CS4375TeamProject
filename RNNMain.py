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

df = pd.read_csv('https://raw.githubusercontent.com/AubreyFeldker/CS4375TeamProject/main/processed_weather_df.csv')

df[df['Cold'].astype(bool)]  

df = df.sort_values(by = 'Timestamp')
df = df.reset_index()

X = df.drop('Precipitation(in)', axis=1)
Y = df['Precipitation(in)']

split_ratio = 0.8
split_index = int(len(df) * split_ratio)

X_train = X[:split_index]
Y_train = Y[:split_index]
X_test = X[split_index:]
Y_test = Y[split_index:]

print(X_train)