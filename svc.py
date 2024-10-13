import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder

from scipy.cluster.hierarchy import\
     (dendrogram,
      cut_tree)
from ISLP.cluster import compute_linkage

#SVM and SVC decision boundaries
data = pd.read_csv("heart.csv")

X=data.iloc[:,:13]
y=data["target"]
SS=StandardScaler()
X_scaled=SS.fit_transform(X)
#PCA
pca=PCA()
X_tf=pca.fit_transform(X_scaled)
pca.explained_variance_ratio_
svc=svm.SVC(kernel="rbf")
svc.fit(X_tf[:,:2],y)
plt.scatter(X_tf[:, 0], X_tf[:, 2], c=y, s=150, edgecolors="k")
plt.show()


fig, ax = plt.subplots(figsize=(4, 3))
# Plot decision boundary and margins
common_params = {"estimator": svc, "X": X_tf[:,:2], "ax": ax}
# DecisionBoundaryDisplay.from_estimator(**common_params,
#             response_method="predict",
#             plot_method="pcolormesh",
#             alpha=0.3,
#         )
DecisionBoundaryDisplay.from_estimator(
            **common_params,
            response_method="auto",
            plot_method="contour",
            levels=[-1, 0, 1],
            colors=["r", "b", "g"],
            linestyles=["--", "-", "--"],
        )
# Plot bigger circles around samples that serve as support vectors
ax.scatter(
            svc.support_vectors_[:, 0],
            svc.support_vectors_[:, 1],
            s=250,
            facecolors="none",
            edgecolors="k",
        )
# Plot samples by color and add legend
scatter = ax.scatter(X_tf[:, 0], X_tf[:, 1], c=y, s=150, edgecolors="k")
ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
ax.set_title(f" Decision boundaries of rbf kernel in SVC")
fig = plt.show()
