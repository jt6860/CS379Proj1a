# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:04:00 2023

@author: John Torres
Course: CS379 - Machine Learning
Project 1a: Clustering The Titanic Dataset
Unsupervised Clustering Algorithm: MeanShift
"""

# Importing Libraries
import pandas as pd 
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing

# Helper function
text_digit_vals = {}

def handle_non_numerical_data(df):
    columns = df.columns.values

    def convert_to_int(val):
        return text_digit_vals[val]

    for column in columns:
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))

    return df

# Read in Excel document and convert to DataFrame.
df = pd.read_excel('CS379T-Week-1-IP.xls', index_col=0)  

# Create a copy of the original DataFrame.
original_df = pd.DataFrame.copy(df)

# Drop 'body' and 'name' columns and fill NaN values with 0.
df.drop(columns=['body','name'], inplace=True)
df.fillna(0, inplace=True)

# Handle non-numerical data
df = handle_non_numerical_data(df)
df.drop(columns=['ticket','home.dest'], inplace=True)

# Preprocess the data
X = np.array(df.drop(columns=['survived']).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

# Perform MeanShift clustering
clf = MeanShift()
clf.fit(X)

# Get cluster labels and centers
labels = clf.labels_
cluster_centers = clf.cluster_centers_

# Add cluster group column to DataFrame
df['cluster_group'] = np.nan

# Assign cluster group to each data point
for i in range(len(X)):
    df.loc[i, 'cluster_group'] = labels[i]

# Calculate survival rates for each cluster
n_clusters_ = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters_):
    temp_df = df[df['cluster_group'] == float(i)]
    survival_cluster = temp_df[temp_df['survived'] == 1]
    if len(temp_df) > 0:
        survival_rate = len(survival_cluster) / len(temp_df)
    else:
        survival_rate = 0
    survival_rates[i] = survival_rate
    
print(survival_rates)