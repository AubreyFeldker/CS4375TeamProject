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

# Read in csv file
df = pd.read_csv("https://raw.githubusercontent.com/AubreyFeldker/CS4375TeamProject/main/Dallas-Fort%20Worth%20Metropolitan%20Area%20Weather%20Events%202016-2021.csv")

# Removing Time and Date Columns since Timestamp covers both
df = df.drop(['Time(UTC)', 'Date'], axis = 1)

# Remove unhelpful data
df = df[df['Severity'] != 'UNK']
df = df[df['Severity'] != 'Other']
df.dropna()

# Using One-Hot Encoding for the precipitation type and severity
# converts categorical data into numerical data
oneHot = OneHotEncoder()

# Using two different arrays to avoid size mismatch
# Perform encoding
arry = oneHot.fit_transform(df[['Type']]).toarray()
feature_labels = oneHot.categories_

arry2= oneHot.fit_transform(df[['Severity']]).toarray()
feature_labels2 = oneHot.categories_

# Flatten out the arrays
feature_labels = np.array(feature_labels).ravel()
feature_labels2 = np.array(feature_labels2).ravel()

# Create new dataframes with the encoded arrays
df1 = pd.DataFrame(arry, columns = feature_labels)
df2 = pd.DataFrame(arry2, columns = feature_labels2)


# Drop type and severity columns to add in new encoded columns
df = df.drop(['Type','Severity'], axis = 1)

# Concat all columns from df, df1, and df2
df = pd.concat([df1, df2, df], axis=1)

# Renameing Precipitation column to avoid confusion
df.rename(columns = {"Precipitation":"Other"}, inplace = True)

# Standardize Precipitation
scaler = StandardScaler()
df[['Precipitation(in)']] = scaler.fit_transform(df[['Precipitation(in)']])

print(df.head())

# Convert Python DateTime to Unix Timestamp / Remove nan values that for some reason the other ways above did not capture
df['Timestamp'] = df['Timestamp'].apply(lambda x: datetime.datetime.strptime(str(x)[0:-2],'%Y%m%d%H%M').timestamp() if str(x) != 'nan' else -1)
df = df.sort_values(by = 'Timestamp')

df.to_csv('processed_weather_df.csv', index=False)