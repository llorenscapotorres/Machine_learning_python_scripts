import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_openml

california = fetch_california_housing()
print("Shape california:", california.data.shape)

ames = fetch_openml(name="house_prices", as_frame=True, parser="auto")
print("Shape Ames:",ames.data.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(california.data, california.target)

from sklearn.linear_model import LinearRegression 

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Score Training Set:", lr.score(X_train, y_train))
print("Score Test Set:", lr.score(X_train, y_train))
print("lr Coeficientes:", lr.coef_)
print("lr Intercept:", lr.intercept_)

from sklearn.linear_model import Ridge

ridge = Ridge(alpha = 0.1).fit(X_train,y_train)

print("Score Ridge Training Set:", ridge.score(X_train, y_train))
print("Score Ridge Test Set:", ridge.score(X_train, y_train))
print("Coeficientes:", ridge.coef_)

from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)

print("Score Lasso Training Set:", lasso.score(X_train, y_train))
print("Score Lasso Test Set:", lasso.score(X_train, y_train))
print("Numero de caracteristicas utilizadas:", np.sum(lasso.coef_ != 0))

lasso001 = Lasso(alpha = 0.01, max_iter = 1000).fit(X_train, y_train)

print("Score Lasso Training Set:", lasso001.score(X_train, y_train))
print("Score Lasso Test Set:", lasso001.score(X_train, y_train))
print("Numero de caracteristicas utilizadas:", np.sum(lasso001.coef_ != 0))