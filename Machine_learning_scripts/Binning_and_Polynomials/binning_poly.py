import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

def make_wave(n_samples=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(-3, 3, size=n_samples)
    y_no_noise = (np.sin(4 * x) + x)
    y = (y_no_noise + rnd.normal(size=len(x))) / 2
    return x.reshape(-1, 1), y

X, y = make_wave(n_samples = 120)
line = np.linspace(-3,3, 1000, endpoint=False).reshape(-1,1)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(min_samples_leaf=3).fit(X,y)
plt.plot(line, reg.predict(line), label="decision tree")

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label = "linear regression")

plt.plot(X[:,0], y, "o", c = "k")
plt.xlabel("Input feature")
plt.ylabel("Regression output")
plt.legend(loc="best")
plt.show()

from sklearn.preprocessing import KBinsDiscretizer

kb = KBinsDiscretizer(n_bins = 10, strategy = "uniform")
kb.fit(X)
print("bin edges: \n", kb.bin_edges_)
X_binned = kb.transform(X)

kb = KBinsDiscretizer(n_bins=10, strategy="uniform", encode="onehot-dense")
kb.fit(X)
X_binned = kb.transform(X)

line_binned = kb.transform(line)

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label = "linear regression binned")

reg = DecisionTreeRegressor(min_samples_split = 3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label = "decision tree binned")

plt.plot(X[:,0], y, "o", c="k")
plt.vlines(kb.bin_edges_[0], -3,3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.show()

X_combined = np.hstack([X, X_binned])

reg = LinearRegression().fit(X_combined, y)

line_combined = np.hstack([line, line_binned])

plt.plot(line, reg.predict(line_combined), label = "linear regression combined")
plt.vlines(kb.bin_edges_[0], -3,3, linewidth = 1, alpha=.2)
plt.legend(loc = "best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.plot(X[:,0], y, "o", c="k")
plt.show()

X_product = np.hstack([X_binned, X*X_binned])

reg = LinearRegression().fit(X_product, y)

line_product = np.hstack([line_binned, line*line_binned])
plt.plot(line, reg.predict(line_product), label="linear regression product")
plt.vlines(kb.bin_edges_[0], -3,3, linewidth=1, alpha=.2)
plt.plot(X[:,0], y, "o", c="k")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc = "best")
plt.show()

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 10, include_bias = False)
poly.fit(X)
X_poly = poly.transform(X)

reg = LinearRegression().fit(X_poly, y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label = "polynomial linear regression")
plt.plot(X[:,0], y, "o", c="k")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc = "best")
plt.show()

from sklearn.svm import SVR

for gamma in [1,10]:
    svr = SVR(gamma = gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label = "SVR gamma={}".format(gamma))
plt.plot(X[:,0], y, "o", c = "k")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc = "best")
plt.show()