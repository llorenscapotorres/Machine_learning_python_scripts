import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 0)

from sklearn.svm import SVC

svc = SVC(kernel = "rbf", C = 1000) #valores por defecto C = 1, gamma = 1/n_features
svc.fit(X_train, y_train)

print("Precisión en training:{:.3f}".format(svc.score(X_train, y_train)))
print("Precisión en test:{:.3f}".format(svc.score(X_test, y_test)))

#hay mucha diferencia en la escala de los datos, y por tanto puede fallar este modelo

plt.boxplot(X_train, manage_ticks=False)
plt.yscale("symlog")
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.show()

#escalaremos los datos a mano, pero se puede hacer con una funcion

min_on_training = X_train.min(axis = 0)
range_on_training = (X_train - min_on_training).max(axis = 0)
X_train_scaled = (X_train - min_on_training)/range_on_training
X_test_scaled = (X_test - min_on_training)/range_on_training

svc_scaled = SVC()
svc_scaled.fit(X_train_scaled, y_train)

print("Precisión en training con datos escalados:{:.3f}".format(svc_scaled.score(X_train, y_train)))
print("Precisión en test con datos escalados:{:.3f}".format(svc_scaled.score(X_test, y_test)))




