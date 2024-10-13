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

NCI60 = load_data('NCI60')
nci_labels = NCI60['labels']
nci_data = NCI60['data']
# print(nci_data)
# print(nci_labels)
print(nci_data.shape)
print(nci_labels.value_counts())

# PCA on NCI60

scaler = StandardScaler()
nci_scaled = scaler.fit_transform(nci_data)
nci_pca = PCA()
nci_scores = nci_pca.fit_transform(nci_scaled)
print(f"Variances: {nci_pca.explained_variance_}")
explained_variance = nci_pca.explained_variance_ratio_.cumsum()
print(f"Explain Variance Ratio: {nci_pca.explained_variance_ratio_}")
no_of_components = nci_pca.n_components_
ticks = np.arange(nci_pca.n_components_) + 1
plt.plot(ticks, nci_pca.explained_variance_ratio_.cumsum(), marker="o")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("Cumulative Variance Explained by PC")
plt.show()
# This plot gives us an idea as to how much each PC is contributing towards the explained variance.

# We now plot the first few principal component score vectors, in order to
# visualize the data. The observations (cell lines) corresponding to a given
# cancer type will be plotted in the same color, so that we can see to what
# extent the observations within a cancer type are similar to each other.

cancer_types = list(np.unique(nci_labels))
print(cancer_types)
print(nci_labels)
nci_groups = np.array([cancer_types.index(labels) for labels in nci_labels.values])
print(nci_groups)
fig, axes = plt.subplots(1, 2)

ax1 = axes[0]
ax1.scatter(nci_scores[:, 0], nci_scores[:, 1], c=nci_groups, marker='o', s=50)
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")

ax2 = axes[1]
ax2.scatter(nci_scores[:, 0], nci_scores[:, 2], c=nci_groups, marker='o', s=50)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC3')

plt.title("First 3 PC for the data")

plt.show()

H_clust = AgglomerativeClustering


# Hierarchical Clustering on NCI dataset:
def plot_nci_hc(linkage, cut=-np.inf):
    cargs = {'above_threshold_color': 'black',
             'color_threshold': cut}
    hc = H_clust(n_clusters=None, distance_threshold=0, linkage=linkage.lower()).fit(nci_scaled)
    linkage1 = compute_linkage(hc)
    dendrogram(linkage1, labels=np.asarray(nci_labels), leaf_font_size=10, **cargs)
    plt.title(f"{linkage} Linkage")
    plt.axhline(cut, c='r', linewidth=2)
    plt.show()
    return hc


hc_comp = plot_nci_hc('Complete')

hc_avg = plot_nci_hc('Average')

hc_single = plot_nci_hc('Single')

# We can cut the dendrogram at the height that will yield a particular number of clusters, say four:
plot_nci_hc('Complete', cut=140)


linkage_comp = compute_linkage(hc_comp)
comp_cut = cut_tree(linkage_comp, n_clusters=4).reshape(-1)
print(pd.crosstab(nci_labels['label'], pd.Series(comp_cut.reshape(-1), name='Complete')))

nci_kmeans = KMeans(n_clusters=4, random_state=0, n_init=20).fit(nci_scaled)
print(pd.crosstab(pd.Series(comp_cut, name='HClust '), pd.Series(nci_kmeans.labels_, name='K-means ')))

# Rather than performing hierarchical clustering on the entire data matrix, we can also perform hierarchical clustering on the first few principal
# component score vectors, regarding these first few components as a less
# noisy version of the data.

hc_pca = H_clust(n_clusters=None, distance_threshold=0, linkage='complete').fit(nci_scores[:, :5])
linkage_pca = compute_linkage(hc_pca)
fig, ax = plt.subplots(figsize=(8, 8))
dendrogram(linkage_pca, labels=np.asarray(nci_labels), leaf_font_size=10, ax=ax, above_threshold_color='black', color_threshold=100)
ax.set_title("Hier. Clust. on First Five Score Vectors")
plt.show()

pca_labels = pd.Series(cut_tree(linkage_pca, n_clusters=4).reshape(-1), name='Complete-PCA')
print(pd.crosstab(nci_labels['label'], pca_labels))




