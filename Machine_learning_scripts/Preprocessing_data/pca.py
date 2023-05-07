import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#Vamos a dibujar 30 histogramas para visualizar el dataset (tarda mucho en ejecutarse)

"""fig, axes = plt.subplots(15, 2, figsize = (10,20))

malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

ax = axes.ravel()

for i in range(30):
	_, bins = np.histogram(cancer.data[:, i], bins = 50)
	ax[i].hist(malignant[:,i], bins = bins, color = "Red", alpha = .5)
	ax[i].hist(benign[:,i], bins = bins, color = "Blue", alpha = .5)
	ax[i].set_title(cancer.feature_names[i])
	ax[i].set_yticks(()) 
	ax[0].set_xlabel("Feature Magnitude")
	ax[0].set_ylabel("Frequency")
	ax[0].legend(["malignant", "benign"], loc = "best")
	fig.tight_layout()	
plt.show() """

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

#dibujamos el grafico del dataset cancer en tan solo 2 dimensiones

mask = (cancer.target == 0)

plt.figure(figsize=(8,8))
plt.scatter(X_pca[mask,0], X_pca[mask,1], color = "blue", s = 60, label = "malignant")
plt.scatter(X_pca[~mask,0], X_pca[~mask,1], color = "red", s = 60, marker = "^", label = "benign")
plt.legend(cancer.target_names, loc = "best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()

print("PCA component shape:{}".format(pca.components_.shape))
print(pca.components_)

#dibujaremos un heatmap para visualizar mejor los coeficientes que nos da la funcion anterior

plt.matshow(pca.components_, cmap = "viridis") #heatmap
plt.yticks([0,1],["First component", "Second component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation = 60, ha = "left")
plt.xlabel("Feature")
plt.ylabel("Principal components")
plt.show()