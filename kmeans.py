import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.cluster import KMeans


def buildClusters():
    dataframe = pd.read_csv("Mall_Customers.csv")
    x = dataframe.iloc[:, [3,4]].values # Extract 2 specified columns
    k_means = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 69) # random state needs to be the same, n_clusters from runKmeans()
    y_kmeans = k_means.fit_predict(x) # x,y coordinates of each sample

    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Balanced')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Big-Spender')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Saver')
    plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'yellow', label = 'Reckless-Spender')
    plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'cyan', label = 'Low-Spender')
    plt.scatter(k_means.cluster_centers_[:,0], k_means.cluster_centers_[:,1], s = 50, c = "black", label = "Centroid")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending score")
    plt.legend()
    plt.show()

def runKmeans():
    dataframe = pd.read_csv("Mall_Customers.csv")
    x = dataframe.iloc[:, [3,4]].values # Extract 2 specified columns 
    # print(x.shape)
    # print(x)
    wcss_score = [] # We want this to be as small as possible, and minimum number of clusters
    for i in range(1,11):
        k_means = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 69)
        k_means.fit(x)
        wcss_score.append(k_means.inertia_) # calculate the sum of the distances of all the points from the cluster centroid 
    # plot graphs
    plt.plot(range(1,11), wcss_score)
    plt.title("K-means")
    plt.xlabel("N0. of Clusters")
    plt.ylabel("Score")
    plt.show() 
    buildClusters()

runKmeans()