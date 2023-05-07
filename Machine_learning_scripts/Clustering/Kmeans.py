import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

#ejemplo de que el cluster K-means no suele funcionar para conjuntos de datos con una estructura anomala

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

from matplotlib.colors import LinearSegmentedColormap

colors = ['#0000FF', '#FF0000']
cmap = LinearSegmentedColormap.from_list('cm2', colors, N=2)

X, y = make_moons(n_samples = 200, noise = 0.05, random_state = 0)

kmeans = KMeans(n_clusters = 2, n_init = 10)
kmeans.fit(X)
y_pred = kmeans.predict(X) #sera o 0 o 1, porque tenemos dos clusters

plt.scatter(X[:, 0], X[:, 1], c = y_pred, cmap = cmap, s = 60, edgecolor = "k")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker = "^", c = [cmap(0), cmap(1)], s = 100, linewidth=2, edgecolor = "k")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

kmeans = KMeans(n_clusters = 10, n_init = 10, random_state = 0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c = y_pred, cmap = "Paired", s = 60, edgecolor = "k")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 60, 
	marker = "^", c = range(kmeans.n_clusters), linewidth = 2, cmap="Paired")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
print("Cluster memberships:\n{}".format(y_pred))
plt.show()

distance_features = kmeans.transform(X)
print("Distance feature shape:{}".format(distance_features.shape))
print("Distance features:\n{}".format(distance_features))