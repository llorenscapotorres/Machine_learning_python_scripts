import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles

X, y = make_circles(noise = 0.25, factor = 0.5, random_state = 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

clf = GradientBoostingClassifier(random_state = 0)
clf.fit(X_train, y_train)

print("Decision function:", clf.decision_function(X_test)[:6])

# Crear meshgrid para dibujar la región de decisión
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100),
                     np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Dibujar gráfico de la región de decisión
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Grafico de region de decision
axs[0].contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
axs[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdBu', edgecolor='white', s=50, marker='o')
axs[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdBu', edgecolor='white', s=50, marker='^')
axs[0].set_title("Región de decisión")

# Gráfico de predict_proba
probs = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
probs = probs.reshape(xx.shape)
im = axs[1].imshow(probs, extent=[xx.min(), xx.max(), yy.min(), yy.max()], aspect='auto', origin='lower', cmap='RdBu')
axs[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdBu', edgecolor='white', s=50, marker='o')
axs[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdBu', edgecolor='white', s=50, marker='^')
axs[1].set_title("Probabilidades")
fig.colorbar(im, ax=axs[1], orientation='vertical')

# leyenda
handles = []
handles.append(plt.Line2D([], [], color='blue', marker='^', linestyle='None',
                          markersize=8, label='Test 0'))
handles.append(plt.Line2D([], [], color='red', marker='^', linestyle='None',
                          markersize=8, label='Test 1'))
handles.append(plt.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=8, label='Train 0'))
handles.append(plt.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=8, label='Train 1'))
plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(-0.3, 1.2),
           ncol=4, fancybox=True, shadow=True)

plt.show()