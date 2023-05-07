import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


X, y = make_moons(n_samples = 100, noise = 0.25, random_state = 3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)

#neural network con 10 unidades ocultas, en 2 capas oculta y utilizando la funcion tanh, con aplha = 0.0001
mlp = MLPClassifier(max_iter = 10000, solver = "lbfgs", random_state = 0, activation = "tanh",
	hidden_layer_sizes = [10, 10]).fit(X_train, y_train)

mask = (y == 0)

colors = {0:"blue", 1:"red"}

plt.scatter(X[mask, 0], X[mask, 1], color=colors[0], s=60)
plt.scatter(X[~mask, 0], X[~mask, 1], color=colors[1], s=60, marker="^")
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')

# Generar datos para la gráfica
xx, yy = np.meshgrid(np.linspace(min(X[:,0])-1, max(X[:,0])+1, 100), np.linspace(min(X[:,1])-1, max(X[:,1])+1, 100))
Z = mlp.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Graficar los datos de prueba y la predicción del modelo
plt.contour(xx, yy, Z, levels=[0.5], colors='k')
plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), alpha=0.2, cmap=plt.cm.coolwarm)
plt.show()

print("Precisión en training data:{:.3f}".format(mlp.score(X_train, y_train)))
print("Precisión en test data:{:.3f}".format(mlp.score(X_test, y_test)))

#heatmap de los pesos de los inputs con las 10 unidades ocultas de la primera capa
plt.figure(figsize=(20,5))
plt.imshow(mlp.coefs_[0], interpolation = "none", cmap = "viridis")
plt.yticks(range(2), ["Feature 0", "Feature 1"])
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()