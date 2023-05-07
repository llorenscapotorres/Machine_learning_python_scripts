import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.datasets import load_iris

iris_dataset = load_iris()

print('Las llaves del data iris son:\n', iris_dataset.keys())

print('Variables de perfil:\n', iris_dataset['target_names'])

print('El data frame es en sus 5 primeras filas:\n', iris_dataset['data'][:5])

print('Y su tipo es:', type(iris_dataset['data']))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

iris_dataframe = pd.DataFrame(X_train, columns = iris_dataset.feature_names)

print('El DataFrame creado con pandas es:\n', iris_dataframe)

pd.plotting.scatter_matrix(iris_dataframe, c = y_train, 
	figsize = (15,15), marker = 'o', hist_kwds = {'bins':20}, s = 60, alpha = .8)
#plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)

#Introducimos un nuevo iris

X_new = np.array([[5,2.9,1,0.2]])
prediction = knn.predict(X_new)
print('Prediccion:', iris_dataset['target_names'][prediction])

#Empezamos de verdad

y_pred = knn.predict(X_test)
print('La prediction es:\n', y_pred)
print('Prediccion de los iris del test:\n', iris_dataset['target_names'][y_pred])
print("Valores reales de los iris del test:\n", iris_dataset.target_names[y_test])
print("El indice en donde son diferentes es:", np.where(iris_dataset.target_names[y_pred] != iris_dataset.target_names[y_test]))
print('Valor de la Prediccion:', knn.score(X_test,y_test))
print('Valor de la Prediccion calculado:', np.mean(y_pred == y_test))