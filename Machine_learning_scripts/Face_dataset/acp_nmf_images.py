import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

#cada imagen tiene 87x65 pixels, tiene 3023 fotos de 62 personas diferentes

fig, axes = plt.subplots(2,5, figsize=(15,8), subplot_kw = {"xticks": (), "yticks": ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
	ax.imshow(image)
	ax.set_title(people.target_names[target])
plt.show()

counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
	print("{0:25} {1:3}".format(name, count), end="   ")
	if (i+1)%3 == 0:
		print()

#tomamos solo 50 imagenes por persona para evitar el sesgo

mask = np.zeros(people.target.shape, dtype = np.bool_)
for target in np.unique(people.target):
	mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people = X_people / 255. #escalamos

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify = y_people, random_state = 0)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
print()
print("Test set score of 1-kn:{:.2f}".format(knn.score(X_test, y_test)))
#nos da una precision del 33%, solo identificara correctamente 1 de cada 3 personas

from sklearn.decomposition import PCA

pca = PCA(n_components = 100, whiten=True, random_state = 0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

knn_pca = KNeighborsClassifier(n_neighbors = 1)
knn_pca.fit(X_train_pca, y_train)

print("Test set accuarcy:{:.2f}".format(knn_pca.score(X_test_pca, y_test)))

fig, axes = plt.subplots(3,5, figsize=(15,12), subplot_kw = {"xticks":(), "yticks":()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
	ax.imshow(component.reshape(image_shape), cmap = "viridis")
	ax.set_title("{}.component".format(i+1))
plt.show()

from sklearn.decomposition import NMF #otra forma de reducir dimensionalidad

nmf = NMF(n_components = 15, random_state = 0, max_iter = 400)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

fig, axes = plt.subplots(3,5, figsize = (15,12), subplot_kw = {"xticks": (), "yticks": ()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
	ax.imshow(component.reshape(image_shape))
	ax.set_title("{}.component".format(i))
plt.show()

#ahora vamos a ver cuales son las imagenes que tienen un gran coeficiente en la componente 3 y 7

compn = 3
inds = np.argsort(X_train_nmf[:,compn])[::-1]
fig, axes = plt.subplots(2,5, figsize = (15,8), subplot_kw = {"xticks": (), "yticks": ()})
fig.suptitle("Large component 3")
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
	ax.imshow(X_train[ind].reshape(image_shape))
plt.show()

compn = 7
inds = np.argsort(X_train_nmf[:,compn])[::-1]
fig, axes = plt.subplots(2,5, figsize = (15,8), subplot_kw = {"xticks": (), "yticks": ()})
fig.suptitle("Large component 3")
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
	ax.imshow(X_train[ind].reshape(image_shape))
plt.show()