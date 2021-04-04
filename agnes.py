# Agglomerative Nesting
# It will assume the whole dataset as one cluster and start breaking it into smaller clusters
# Its a bottom up approach 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns


def buildClusters():
    dataframe = pd.read_csv("Mall_Customers.csv")
    x = dataframe.iloc[:, [3,4]].values # Extract 2 specified columns
    dist = linkage(x, "ward")
    dataset_new = dataframe.copy()
    dataset_new['3_clusters'] = fcluster(dist, 3, criterion="maxclust")
    sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=dataset_new, hue="3_clusters")
    plt.title("AGNES clusters")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending score")
    plt.legend()
    plt.show()

def Dendrogram(): 
    dataframe = pd.read_csv("Mall_Customers.csv")
    x = dataframe.iloc[:, [3,4]].values # Extract 2 specified columns
    link = linkage(x, "ward") # calculate nearest point distance 
    dendrogram(link, orientation="top", distance_sort="descending", show_leaf_counts=True)
    plt.title("Dendogram")
    plt.show()
    buildClusters()

Dendrogram()

"""
Dendogram shows us how many optimal number of clusters should be formed
"""