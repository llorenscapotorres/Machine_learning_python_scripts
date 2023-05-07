import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples = 100, noise = 0.25, random_state = 3)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)

forest = RandomForestClassifier(n_estimators = 5, random_state = 2)
forest.fit(X_train, y_train)

from sklearn.tree import export_graphviz
import graphviz
from graphviz import Source

#FORMA MANUAL DE DIBUJAR LOS ARBOLES DEL RANDOM FOREST

#def plot_tree_manual(tree):
#    dot_data = export_graphviz(tree, out_file=None, 
#                               class_names=["0", "1"], 
#                               feature_names=["feature0", "feature1"], 
#                               impurity=False, filled=True)
#    graph = graphviz.Source(dot_data)
#    graph.render(view=True)

#for i, tree in enumerate(forest.estimators_):
#    export_graphviz(tree, out_file = "tree" + str(i) + ".dot", class_names = ["0", "1"], 
#        feature_names = ["feature0", "feature1"], impurity = False, filled = True)
#    with open("tree" + str(i) + ".dot") as f:
#        dot_graph = f.read()
#    graph = graphviz.Source(dot_graph)
#    graph.format = "png"
#    graph.render("tree" + str(i), view = True)

#FORMA CON FUNCION "PLOT_TREE" PARA DIBUJAR LOS ARBOLES DEL RANDOM FOREST

trees = forest.estimators_ #estan todos los arboles aqui dentro
fig, axes = plt.subplots(nrows = 1, ncols = len(trees), figsize = (20, 5))

from sklearn.tree import plot_tree

for i in range(len(trees)):
    ax = axes[i]
    ax.set_axis_off()
    tree = trees[i]
    plot_tree(tree, ax = ax, filled = True)

plt.show()