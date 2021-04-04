import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

dataframe = pd.read_csv("Mall_Customers.csv")
x = dataframe.iloc[:, [3,4]].values # Extract 2 specified columns 
dbscan = DBSCAN(eps=5.5, min_samples=4)
labels=dbscan.fit_predict(x)
# dbscan.fit(x)

# pca = PCA(n_components=2).fit(x)
# pca_2d = pca.transform(x)
# print(pca_2d.shape[0])
# for i in range(0, pca_2d.shape[0]):
#     if dbscan.labels_[i] == 0:
#         c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
#     elif dbscan.labels_[i] == 1:
#         c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
#     elif dbscan.labels_[i] == -1:
#         c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b', marker='*')

print(np.unique(labels))

plt.scatter(x[labels==-1,0], x[labels==-1,1], s=100, c='black')
plt.scatter(x[labels==0,0], x[labels==0,1], s=100, c='blue')
plt.scatter(x[labels==1,0], x[labels==1,1], s=100, c='red')
plt.scatter(x[labels==2,0], x[labels==2,1], s=100, c='green')
plt.scatter(x[labels==3,0], x[labels==3,1], s=100, c='yellow')
plt.scatter(x[labels==4,0], x[labels==4,1], s=100, c='pink')
plt.scatter(x[labels==5,0], x[labels==5,1], s=100, c='cyan')



# plt.legend([c1, c2, c3, c4], ['Cluster 1', 'Cluster 2', 'Cluster 3','Noise'])
plt.title('DBSCAN')
plt.xlabel("Annual Income")
plt.ylabel("Spending score")
plt.legend()
plt.show()