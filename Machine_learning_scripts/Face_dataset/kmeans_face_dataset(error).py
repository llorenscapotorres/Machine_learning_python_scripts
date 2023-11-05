import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

#tomamos solo 50 imagenes por persona para evitar el sesgo

mask = np.zeros(people.target.shape, dtype = np.bool_)
for target in np.unique(people.target):
	mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people = X_people / 255. #escalamos

from sklearn.decomposition import PCA

pca = PCA(n_components = 100, whiten = True, random_state = 0)
X_pca = pca.fit_transform(X_people)

km = KMeans(n_clusters = 10, random_state = 0, n_init = 10)
labels_km = km.fit_predict(X_pca)
print("Cluster sizes k_means: {}".format(np.bincount(labels_km)))

#as we clustered in the representation produced by PCA, we need to rotate the cluster centers back
# into the original space to visualize them, using pca.inverse_transform
fig, axes = plt.subplots(2,5,subplot_kw={"xticks":(),"yticks":()}, figsize=(12,4))
for center, ax in zip(km.cluster_centers_, axes.ravel()):
	ax.imshow(pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax = 0)
plt.show()

def plot_kmeans_faces(km, pca, X_pca, X_people, y_people, target_names):
    n_clusters = 10
    image_shape = (87, 65)
    fig, axes = plt.subplots(n_clusters, 11, subplot_kw={'xticks': (), 'yticks': ()},
                             figsize=(10, 15), gridspec_kw={"hspace": .3})

    for cluster in range(n_clusters):
        center = km.cluster_centers_[cluster]
        mask = km.labels_ == cluster
        dists = np.sum((X_pca - center) ** 2, axis=1)
        dists[~mask] = np.inf
        inds = np.argsort(dists)[:5]
        dists[~mask] = -np.inf
        inds = np.r_[inds, np.argsort(dists)[-5:]]
        axes[cluster, 0].imshow(pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax=1)
        for image, label, asdf, ax in zip(X_people[inds], y_people[inds],
                                          km.labels_[inds], axes[cluster, 1:]):
            ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
            ax.set_title("%s" % (target_names[label].split()[-1]), fontdict={'fontsize': 9})

    # add some boxes to illustrate which are similar and which dissimilar
    rec = plt.Rectangle([-5, -30], 73, 1295, fill=False, lw=2)
    rec = axes[0, 0].add_patch(rec)
    rec.set_clip_on(False)
    axes[0, 0].text(0, -40, "Center")

    rec = plt.Rectangle([-5, -30], 385, 1295, fill=False, lw=2)
    rec = axes[0, 1].add_patch(rec)
    rec.set_clip_on(False)
    axes[0, 1].text(0, -40, "Close to center")

    rec = plt.Rectangle([-5, -30], 385, 1295, fill=False, lw=2)
    rec = axes[0, 6].add_patch(rec)
    rec.set_clip_on(False)
    axes[0, 6].text(0, -40, "Far from center")

plot_kmeans_faces(km, pca, X_pca, X_people, y_people, people.target_names)
plt.show()