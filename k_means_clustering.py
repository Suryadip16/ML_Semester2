import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
np.random.seed(69)
x = np.random.standard_normal((50, 2))
x[:25, 0] += 3
x[:25, 1] -= 4
kmeans = KMeans(n_clusters=2, random_state=2, n_init=20).fit(x)
print(kmeans.labels_)

fig, ax = plt.subplots(1, 1)
ax.scatter(x[:, 0], x[:, 1], c=kmeans.labels_)
ax.set_title("K-Means Clustering Results with K = 2")
plt.show()

