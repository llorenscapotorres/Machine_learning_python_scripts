import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.datasets import make_blobs

X, y = make_blobs(centers = 4, random_state = 8)
y = y % 2

#dibujamos el scatter plot y la prediccion lineal

cond0 = (y == 0)
cond1 = (y == 1)

colors = {0:"blue", 1:"red"}

plt.scatter(X[cond0, 0], X[cond0, 1], color=colors[0], s=30, label='Target 0')
plt.scatter(X[cond1, 0], X[cond1, 1], color=colors[1], s=30, marker="^",label='Target 1')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend()

from sklearn.svm import LinearSVC

linear_svm = LinearSVC(max_iter = 10000).fit(X,y)

coef = linear_svm.coef_[0]
intercept = linear_svm.intercept_[0]
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
x_plot = np.linspace(x_min, x_max)
y_plot = (-coef[0] / coef[1]) * x_plot - intercept / coef[1]
plt.plot(x_plot, y_plot, 'k-')

plt.show()

#añadimos una nueva variable no linear, y hacemos la prediccion con un plano

X_new = np.hstack([X, X[:,1:]**2])

linear_svm_3d = LinearSVC(max_iter = 10000).fit(X_new, y)
coef_3d, intercept_3d = linear_svm_3d.coef_[0], linear_svm_3d.intercept_

from mpl_toolkits.mplot3d import Axes3D, axes3d

figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')

xx = np.linspace(X_new[:,0].min()-2, X_new[:,0].max()+2, 50)
yy = np.linspace(X_new[:,1].min()-2, X_new[:,1].max()+2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef_3d[0]*XX + coef_3d[1]*YY + intercept_3d) / -coef_3d[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha = 0.3)

mask = (y == 0)
ax.scatter(X_new[mask,0], X_new[mask,1], X_new[mask,2], c = "b", cmap = "bwr", s = 60, edgecolor = "k")
ax.scatter(X_new[~mask,0], X_new[~mask,1], X_new[~mask,2], c = "r", cmap = "bwr", s = 60, edgecolor = "k")
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")
plt.show()

#proyectamos el plano en dos dimensiones

ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels = [dec.min(), 0, dec.max()], cmap = "bwr", alpha = 0.5)
plt.scatter(X[cond0, 0], X[cond0, 1], color=colors[0], s=30, label='Target 0')
plt.scatter(X[cond1, 0], X[cond1, 1], color=colors[1], s=30, marker="^",label='Target 1')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

import pandas as pd

print(pd.__version__)
