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


def usarrests_pca():
    usa_arrests_df = pd.read_csv("USArrests.csv")
    usarrests_vars = usa_arrests_df.var()
    print(usarrests_vars)
    print(usa_arrests_df)
    names = usa_arrests_df["rownames"]
    usa_arrests = usa_arrests_df.drop(["rownames"], axis=1)

    # PCA looks for derived variables
    # that account for most of the variance in the data set. If we do not scale the
    # variables before performing PCA, then the principal components would
    # mostly be driven by the Assault variable, since it has by far the largest
    # variance.

    scaler = StandardScaler(with_mean=True, with_std=True)
    usa_arrests_scaled = scaler.fit_transform(usa_arrests)

    pcaUS = PCA()
    pcaUS.fit(usa_arrests_scaled)

    print(f"Means after PCA :{pcaUS.mean_}")

    scores = pcaUS.transform(usa_arrests_scaled)
    components = pcaUS.components_
    labels = usa_arrests.columns
    print(f"Components: {pcaUS.components_}")
    print(f"Scores: {scores}")
    print(f"STD of principal components are as follows: {scores.std(0, ddof=1)}")
    print(f"Variances: {pcaUS.explained_variance_}")
    print(f"Explain Variance Ratio: {pcaUS.explained_variance_ratio_}")

    fig, axes = plt.subplots(1, 2)
    ticks = np.arange(pcaUS.n_components_) + 1
    ax = axes[0]
    ax.plot(ticks, pcaUS.explained_variance_ratio_, marker="o")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Proportion of Variance Explained")
    ax.set_ylim([0, 1])
    ax.set_xticks(ticks)
    ax = axes[1]
    ax.plot(ticks, pcaUS.explained_variance_ratio_.cumsum(), marker="o")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Proportion of Variance Explained")
    ax.set_ylim([0, 1])
    ax.set_xticks(ticks)
    plt.show()

    return scores, components, labels, names


# The biplot is a common visualization method used with PCA. It is not
# built in as a standard part of sklearn, though there are python packages
# that do produce such plots. Here we make a simple biplot manually.

def biplot(score, coeff, names, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    fig, ax = plt.subplots()
    ax.scatter(xs * scalex, ys * scaley)
    for i, txt in enumerate(names):
        ax.annotate(txt, (xs[i] * scalex, ys[i] * scaley))
    for i in range(n):
        ax.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r')
        if labels is None:
            ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color="g", ha="center", va="center")
        else:
            ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color="g", ha="center", va="center")
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))


def main():
    scores, components, labels, names = usarrests_pca()
    biplot(scores[:, 0:2], np.transpose(components[0:2, :]), names, labels=list(labels))
    plt.show()


if __name__ == '__main__':
    main()

# i, j = 0, 1 # which components
# fig, ax = plt.subplots(1, 1)
# ax.scatter(scores[:, 0], scores[:, 1])
# ax.set_xlabel("PC%d" % (i + 1))
# ax.set_ylabel("PC%d" % (j + 1))
#
# for k in range(pcaUS.components_.shape[1]):
#     ax.arrow(0, 0, pcaUS.components_[i, k], pcaUS.components_[j, k])
#     ax.text(pcaUS.components_.shape[1], pcaUS.components_[j, k],
#             usa_arrests.columns[k])
# scale_arrow = s_ = 2
# scores[:, 1] *= -1
# pcaUS.components_[1] *= -1
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.scatter(scores[:, 0], scores[:, 1])
# ax.set_xlabel("PC%d" % (i + 1))
# ax.set_ylabel("PC%d" % (j + 1))
#
#
# for k in range(pcaUS.components_.shape[1]):
#     ax.arrow(0, 0, s_ * pcaUS.components_[i, k], s_ * pcaUS.components_[j, k])
#     ax.text(s_ * pcaUS.components_.shape[1], s_ * pcaUS.components_[j, k],
#             usa_arrests.columns[k])
# plt.show()
