import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 42)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state = 0) #Modelo dejando todas las hojas puras, es a dir, unpruned tree
tree.fit(X_train, y_train)

print("Precisión en training:{:.3f}".format(tree.score(X_train, y_train)))
print("Precisión en test:{:.3f}".format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth = 4, random_state = 0) #Modelo dejando maxima profundidad a 4, asi evita el overfitting
tree.fit(X_train, y_train)

print("Precisión en training con depth = 4:{:.3f}".format(tree.score(X_train, y_train)))
print("Precisión en test con depth = 4:{:.3f}".format(tree.score(X_test, y_test)))

#FORMA MANUAL DE DIBUJAR UN ARBOL

#from sklearn.tree import export_graphviz

#export_graphviz(tree, out_file = "tree.dot", class_names = ["malignant", "bening"], 
#	feature_names = cancer.feature_names, impurity = False, filled = True)

#import graphviz
#from graphviz import Source

#def plot_tree(tree):
#	graph = Source(tree)
#	graph.format = "png"
#	graph.render("tree", view = True)

#with open("tree.dot") as f:
#	dot_graph = f.read()

#plot_tree(dot_graph)

#FORMA CON FUNCION DE DIBUJAR UN ARBOL

from sklearn.tree import plot_tree  

plot_tree(tree, filled = True)
plt.show()

print("Feature importances:\n", tree.feature_importances_)

def plot_feature_importances_cancer(model):
	n_features = cancer.data.shape[1]
	plt.barh(np.arange(n_features), model.feature_importances_, align = "center")
	plt.yticks(np.arange(n_features), cancer.feature_names)
	plt.xlabel("Feature importances")
	plt.ylabel("Features")
	plt.ylim(-1, n_features)
	plt.show()

plot_feature_importances_cancer(tree)

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100, random_state = 0)
forest.fit(X_train, y_train)

print("Precisión training set del forest:{:.3f}".format(forest.score(X_train, y_train)))
print("Precisión test set del forest:{:.3f}".format(forest.score(X_test, y_test)))

plot_tree(forest, filled = True)
plt.show()

plot_feature_importances_cancer(forest)

#print(cancer.target_names[forest.predict(X_test)])

from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(max_depth = 1 ,random_state = 0) #también se puede poner learning_rate = ...
gbrt.fit(X_train,y_train)

print("Precisión training set del GradientBoostingClassifier:{:.3f}".format(gbrt.score(X_train, y_train)))
print("Precisión test set del GradientBoostingClassifier:{:.3f}".format(gbrt.score(X_test, y_test)))

plot_feature_importances_cancer(gbrt)