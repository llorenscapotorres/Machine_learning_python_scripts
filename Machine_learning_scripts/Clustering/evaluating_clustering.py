import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.metrics.cluster import adjusted_rand_score #ARI
from sklearn.datasets import make_moons

X, y = make_moons(n_samples = 200, noise = 0.05, random_state = 0)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize = (15,3), subplot_kw = {"xticks":(), "yticks":()})

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

algorithms = [KMeans(n_clusters = 2, n_init = 10), AgglomerativeClustering(n_clusters = 2), DBSCAN()]

#With ground truth

#create a random cluster assignment for reference

random_state = np.random.RandomState(seed = 0)
random_cluster = random_state.randint(low = 0, high = 2, size = len(X))

axes[0].scatter(X_scaled[:,0], X_scaled[:,1], c = random_cluster, s = 60)
axes[0].set_title("Random assignment - ARI: {:.2f}".format(adjusted_rand_score(y, random_cluster)))

for ax, algorithm in zip(axes[1:], algorithms):
	clusters = algorithm.fit_predict(X_scaled)
	ax.scatter(X_scaled[:,0], X_scaled[:,1], c = clusters, s = 60)
	ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__, adjusted_rand_score(y, clusters)))
plt.show()

#Without ground truth

from sklearn.metrics.cluster import silhouette_score

fig, axes = plt.subplots(1, 4, figsize = (15,3), subplot_kw = {"xticks":(), "yticks":()})

axes[0].scatter(X_scaled[:,0], X_scaled[:,1], c = random_cluster, s = 60)
axes[0].set_title("Random assignment - ARI: {:.2f}".format(silhouette_score(X_scaled, random_cluster)))

for ax, algorithm in zip(axes[1:], algorithms):
	clusters = algorithm.fit_predict(X_scaled)
	ax.scatter(X_scaled[:,0], X_scaled[:,1], c = clusters, s = 60)
	ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__, silhouette_score(X_scaled, clusters)))
plt.show()
