import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Minimos por caracteristica antes de escalar:\n", np.round(X_train.min(axis = 0), 2))
print("Minimos por caracteristica despues de escalar:\n", np.round(X_train_scaled.min(axis = 0), 2))

from sklearn.svm import SVC

svm = SVC(C = 100)
svm.fit(X_train, y_train)

print("Test accuarcy:{:.2f}".format(svm.score(X_test, y_test)))

svm.fit(X_train_scaled, y_train)

print("Test accuarcy scaled:{:.2f}".format(svm.score(X_test_scaled, y_test)))













