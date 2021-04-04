# Agglomerative Nesting
# It will assume the whole dataset as one cluster and start breaking it into smaller clusters
# Its a bottom up approach 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns


def Dendrogram(): 
    
