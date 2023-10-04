from sklearn.tree import DecisionTreeClassifier
from model_trainers.trainer import create_train_validate_model_group
from plot import plot_accuracy_group
from dump import get_dump_file
import numpy as np
import matplotlib.pyplot as plt

def train_sklearn_tree(X, y):
    hyper_parameters = [
        ("gini", "best"), ("entropy", "best"), ("log_loss", "best"),
        ("gini", "random"), ("entropy", "random"), ("log_loss", "random")
        ]

    models = create_train_validate_model_group(
        "sklearn.tree",
        hyper_parameters,
        create_sklearn_tree, 
        X, y)

    models.print_details()

    plot_accuracy_group(models)

    return models

def create_sklearn_tree(hyper_params):
    (criterion, split) = hyper_params
    return f"sklearn.tree-{criterion}-{split}", DecisionTreeClassifier(criterion=criterion, splitter=split)