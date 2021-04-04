import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

dataframe = pd.read_csv("Mall_Customers.csv")
x = dataframe.iloc[:, [3,4]].values # Extract 2 specified columns 
dbscan = DBSCAN(eps=5.5, min_samples=4)
labels=dbscan.fit_predict(x)

print(np.unique(labels))

plt.scatter(x[labels==-1,0], x[labels==-1,1], s=100, c='black')
plt.scatter(x[labels==0,0], x[labels==0,1], s=100, c='blue')
plt.scatter(x[labels==1,0], x[labels==1,1], s=100, c='red')
plt.scatter(x[labels==2,0], x[labels==2,1], s=100, c='green')
plt.scatter(x[labels==3,0], x[labels==3,1], s=100, c='yellow')
plt.scatter(x[labels==4,0], x[labels==4,1], s=100, c='pink')
plt.scatter(x[labels==5,0], x[labels==5,1], s=100, c='cyan')


plt.title('DBSCAN')
plt.xlabel("Annual Income")
plt.ylabel("Spending score")
plt.legend()
plt.show()