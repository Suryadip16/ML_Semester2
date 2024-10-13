import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from scipy.cluster.hierarchy import (dendrogram, cut_tree)
from sklearn.cluster import (KMeans, AgglomerativeClustering)
from ISLP.cluster import compute_linkage

X = np.random.standard_normal((50, 2))
X[:25, 0] += 3
X[:25, 1] += 4

Hclust = AgglomerativeClustering
hc_complete_linkage = Hclust(distance_threshold=0, n_clusters=None, linkage='complete')
hc_avg_linkage = Hclust(distance_threshold=0, n_clusters=None, linkage='average')
hc_single_linkage = Hclust(distance_threshold=0, n_clusters=None, linkage='single')

hc_single_linkage.fit(X)
hc_avg_linkage.fit(X)
hc_complete_linkage.fit(X)

#Calculate Pairwise Euclidean Distances between pair of data points

D = np.zeros((X.shape[0], X.shape[0]))
for i in range(X.shape[0]):
    x = np.multiply.outer(np.ones(X.shape[0]), X[i])
    D[i] = np.sqrt(np.sum((X - x) ** 2, 1))
hc_single_linkage_precomputed = Hclust(distance_threshold=0, n_clusters=None, metric='precomputed', linkage='single')
hc_single_linkage_precomputed.fit(D)

cargs = {'color_threshold': -np.inf,
         'above_threshold_color': 'black'}
linkage_complete = compute_linkage(hc_complete_linkage)
fig, ax = plt.subplots(1, 1)
dendrogram(linkage_complete, ax=ax, **cargs)
#plt.show()

fig, ax = plt.subplots(1, 1)
dendrogram(linkage_complete, ax=ax, color_threshold=2, above_threshold_color='black')
plt.show()

print("Hierarchical Clustering Without Scaled data")
print(cut_tree(linkage_complete, n_clusters=4).T)

# Hclust on scaled data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
hc_complete_linkage_scaled = Hclust(distance_threshold=0, n_clusters=None, linkage="complete").fit(X_scaled)
linkage_complete_scaled = compute_linkage(hc_complete_linkage_scaled)
fig, ax = plt.subplots(1, 1)
dendrogram(linkage_complete_scaled, ax=ax, **cargs)
ax.set_title("Hierarchical Clustering With Scaled data ")
plt.show()
print("Hierarchical Clustering With Scaled data")
print(cut_tree(linkage_complete_scaled, n_clusters=2).T)

# X = np.random.standard_normal((30, 3))
corD = 1 - np.corrcoef(X)
hc_complete_linkage_corr = Hclust(linkage="complete", distance_threshold=0, n_clusters=None, metric="precomputed")
hc_complete_linkage_corr.fit(corD)
linkage_complete_corr = compute_linkage(hc_complete_linkage_corr)
fig, ax = plt.subplots(1, 1)
dendrogram(linkage_complete_corr, ax=ax, **cargs)
ax.set_title("Complete Linkage with Correlation-Based Dissimilarity Metric")
plt.show()
print("Hierarchical Clustering With Correlation Based Dissimilarity Metric")
print(cut_tree(linkage_complete_corr, n_clusters=2).T)


