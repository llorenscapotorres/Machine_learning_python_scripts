import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

X, y = make_blobs(random_state = 1)

agg = AgglomerativeClustering(n_clusters = 3)
assignment = agg.fit_predict(X)

cond0 = (assignment == 0)
cond1 = (assignment == 1)
cond2 = (assignment == 2)

plt.scatter(X[cond0,0], X[cond0,1], color = "blue", s = 60)
plt.scatter(X[cond1,0], X[cond1,1], color = "red", marker = "^", s = 60)
plt.scatter(X[cond2,0], X[cond2,1], color = "green", marker = "v", s = 60)
plt.legend(["Cluster 0", "Cluster 1", "Cluster 2"], loc = "best")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#Vamos a dibujar ahora un dendograma

from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(random_state = 0, n_samples = 12)

linkage_array = ward(X)
dendrogram(linkage_array)

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25,7.25], "--", c = "k")
ax.plot(bounds, [4,4], "--", c = "k")
ax.text(bounds[1], 7.25, "two clusters", va = "center", fontdict={"size":10})
ax.text(bounds[1], 4, "three clusters", va = "center", fontdict={"size":10})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")
plt.show()