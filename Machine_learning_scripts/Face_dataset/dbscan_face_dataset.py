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

#DBSCAN

dbscan = DBSCAN()
labels = dbscan.fit_predict(X_pca)
print("Unique labels: {}".format(np.unique(labels))) #only have noise

dbscan = DBSCAN(min_samples = 3)
labels = dbscan.fit_predict(X_pca)
print("Unique labels: {}".format(np.unique(labels))) #only have noise again

dbscan = DBSCAN(min_samples = 3, eps = 15)
labels = dbscan.fit_predict(X_pca)
print("Unique labels: {}".format(np.unique(labels))) #we get a single cluster [-1, 0]
print("Number of points per cluster: {}".format(np.bincount(labels+1)))

from PIL import Image

noise = X_people[labels == -1] #probablemente estas imagenes esten dañadas, por eso no se muestran correctamente

fig, axes = plt.subplots(3, 9, subplot_kw={"xticks":(), "yticks":()}, figsize=(12,4))
for image, ax in zip(noise, axes.ravel()):
    # Crear una imagen a partir de los datos de la matriz de datos
    img = Image.fromarray((image.reshape(image_shape) * 255).astype(np.uint8))
    # Mostrar la imagen en el eje
    ax.imshow(img)
plt.show()

for eps in [1,3,5,7,9,11,13]:
	print("\neps={}".format(eps))
	dbscan = DBSCAN(eps = eps, min_samples = 3)
	labels = dbscan.fit_predict(X_pca)
	print("Number of clusters: {}".format(len(np.unique(labels))))
	print("Cluster sizes: {}".format(np.bincount(labels+1)))

#the results for eps=7 look most interesting

dbscan = DBSCAN(min_samples=3, eps=7)
labels = dbscan.fit_predict(X_pca)

for cluster in range(max(labels)+1):
	mask = (labels == cluster)
	n_images = np.sum(mask)
	fig, axes = plt.subplots(1, n_images, figsize=(n_images*1.5, 4), 
		subplot_kw={"xticks":(), "yticks":()})
	for image, label, ax in zip(X_people[mask], y_people[mask], axes):
		ax.imshow(image.reshape(image_shape), vmin = 0, vmax = 1)
		ax.set_title(people.target_names[label].split()[-1])
plt.show()