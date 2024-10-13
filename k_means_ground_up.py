import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Randomly initialize cluster centroids
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Assign each data point to the nearest centroid
            clusters = [[] for _ in range(self.n_clusters)]
            for point in X:
                cluster_idx = np.argmin([np.linalg.norm(point - centroid) for centroid in self.centroids])
                clusters[cluster_idx].append(point)

            # Update centroids based on the mean of data points in each cluster
            new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])

            # Check for convergence
            if np.allclose(new_centroids, self.centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        # Predict the cluster for each data point
        return np.argmin([[np.linalg.norm(point - centroid) for centroid in self.centroids] for point in X], axis=1)


# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 2)

# Instantiate and fit KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.predict(X)

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
