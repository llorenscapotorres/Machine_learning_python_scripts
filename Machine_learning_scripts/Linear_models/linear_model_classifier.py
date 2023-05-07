import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print("Cancer keys:\n", cancer.keys())
print("The structure of dataset is:",cancer.data.shape)

maligno = 0
benigno = 0
for i in range(569):
	if cancer.target[i] == 0:
		maligno += 1
	elif cancer.target[i] == 1:
		benigno += 1

print("Maligno:",maligno, "Benigno:",benigno)

print("Sample counts per class:\n",{n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 66)

training_accuarcy = []
test_accuarcy = []

from sklearn.neighbors import KNeighborsClassifier

neighbors_settings = range(1,11)
for n_neighbors in neighbors_settings:
	clf = KNeighborsClassifier(n_neighbors = n_neighbors)
	clf.fit(X_train, y_train)
	training_accuarcy.append(clf.score(X_train, y_train))
	test_accuarcy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuarcy, label = "training accuarcy")
plt.plot(neighbors_settings, test_accuarcy, label = "test accuarcy")
plt.ylabel("Accuarcy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

logreg = LogisticRegression(C = 1, max_iter = 100000).fit(X_train, y_train)

print("Accuarcy training set con C = 1: {:.3f}".format(logreg.score(X_train, y_train))) 
print("Accuarcy test set con C = 1: {:.3f}".format(logreg.score(X_test, y_test))) 

logreg100 = LogisticRegression(C = 100, max_iter = 100000).fit(X_train, y_train)

print("Accuarcy training set con C = 100: {:.3f}".format(logreg100.score(X_train, y_train))) 
print("Accuarcy test set con C = 100: {:.3f}".format(logreg100.score(X_test, y_test)))

#print("Los coeficientes para C = 1 y C = 100 son:", logreg.coef_.T, logreg100.coef_.T)

for C in [0.001, 100, 10000]:
	lr_l1 = LogisticRegression(C = C, penalty = "l1", solver = "saga", max_iter = 100000).fit(X_train, y_train)
	print("Training accuarcy of L1 logreg with C =", C ,"{:.2f}".format(lr_l1.score(X_train,y_train)))
	print("Test accuarcy of L1 logreg with C =", C ,"{:.2f}".format(lr_l1.score(X_test,y_test)))

#L1 regularization hace algunas características igual a 0, como en Lassos de linear_regression