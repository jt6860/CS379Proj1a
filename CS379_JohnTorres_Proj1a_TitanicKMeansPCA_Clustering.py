# -*- coding: utf-8 -*-
"""
Created on Tues Jan 2 15:04:00 2023

Author: John Torres
Course: CS379 - Machine Learning
Project 1a: Clustering The Titanic Dataset
Unsupervised Clustering Algorithm: K-Means with Principal Component Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

def preprocess_data():
    # Read the Excel document and convert it to a DataFrame for preprocessing.
    df = pd.read_excel('CS379T-Week-1-IP.xls', index_col=0)
    df = df.reset_index(names=['pclass'])

    # Drop unused columns.
    drop_columns = ['body', 'name', 'ticket', 'home.dest', 'cabin']
    df.drop(columns = drop_columns, inplace = True)

    # Fill NaN values with specific values
    new_values = {"boat": "X", "embarked": "X"}
    df.fillna(value = new_values, inplace = True)
    df.fillna(0, inplace = True)

    # Encoding for non-numerical columns
    non_numerical_columns = ['sex', 'embarked', 'boat']
    encoder = LabelEncoder()
    for column in non_numerical_columns:
        df[column] = encoder.fit_transform(df[column].astype('str'))

    return df

def perform_pca(df):
    # Perform dimensionality reduction using PCA
    reduction_pca = PCA(n_components=3)
    reduced_features = reduction_pca.fit_transform(df)
    return reduced_features

def perform_clustering(reduced_features):
    # Perform clustering using KMeans
    kmeans = KMeans(n_clusters=8, n_init=50)
    clustKM = kmeans.fit(reduced_features)
    return clustKM

def visualize_clusters(reduced_features, clustKM):
    # Visualize the clusters
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clustKM.labels_, label='Datapoints')
    plt.scatter(clustKM.cluster_centers_[:, 0], clustKM.cluster_centers_[:, 1], c='red', label='KMeans Clusters')
    plt.title("KMeans Results")
    plt.legend()
    plt.show()

def calculate_survival_rates(df, clustKM):
    # Get cluster labels and centers
    labels = clustKM.labels_

    # Add cluster group column to DataFrame
    df['cluster_group'] = labels

    # Calculate survival rates for each cluster
    survival_rates = df.groupby('cluster_group')['survived'].mean()

    return survival_rates

def calculate_mean_survival_rate(survival_rates):
    # Calculate the mean survival rate
    mean_survival_rate = survival_rates.mean()
    return mean_survival_rate


def main():
    # Main Workflow function
    df = preprocess_data()
    reduced_features = perform_pca(df)
    clustKM = perform_clustering(reduced_features)
    visualize_clusters(reduced_features, clustKM)
    survival_rates = calculate_survival_rates(df, clustKM)
    mean_survival_rate = calculate_mean_survival_rate(survival_rates)

    print("\nSurvival Rates: ")
    print(survival_rates)

    print("The mean survival rate: " + str(mean_survival_rate))

main()