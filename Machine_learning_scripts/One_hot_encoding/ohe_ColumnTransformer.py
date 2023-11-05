import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

with open('data/adult.data', 'r') as file:
    datos = file.read()

# Convertir a lista de filas
filas = datos.split('\n')[:-1]  # Eliminar última fila en blanco

# Convertir a lista de listas de valores
datos_lista = [fila.split(', ') for fila in filas]

# Crear DataFrame con nombres de columnas personalizados
columnas = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
            "relationship", "race", "gender", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
data = pd.DataFrame(datos_lista, columns=columnas)
data = data[["age", "workclass", "education", "gender", "hours-per-week", "occupation", "income"]]

# Imprimir DataFrame de manera legible
print(data.head().to_string(index=False, col_space=15, justify="left"))

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("scaling", StandardScaler(), ["age", "hours-per-week"]), 
    ("onehot", OneHotEncoder(sparse_output = False), 
        ["workclass", "education", "gender", "occupation"])])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data_features = data.drop("income", axis = 1)
X_train, X_test, y_train, y_test = train_test_split(data_features, data.income, random_state = 0)

X_train.dropna(inplace=True)
X_test.dropna(inplace=True)
y_train.dropna(inplace=True)
y_test.dropna(inplace=True)

ct.fit(X_train)
X_train_trans = ct.transform(X_train)

logreg = LogisticRegression(max_iter = 200)
logreg.fit(X_train_trans, y_train)

X_test_trans = ct.transform(X_test)

print("Test score: {:.2f}".format(logreg.score(X_test_trans, y_test)))