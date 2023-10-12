from project1.interface import P1DecisionTreeClassifier
from model_trainers.trainer import create_train_validate_model_group
from plot import plot_accuracy_group

def train_p1_tree(X, y):
    hyper_parameters = [
        ("entropy", False),
        ("gini", False),
        ("entropy", True),
        ("gini", True)
    ]

    models = create_train_validate_model_group(
        "p1.tree",
        hyper_parameters,
        create_p1_tree, 
        X, y)

    models.print_details()

    plot_accuracy_group(models)

    return models

def create_p1_tree(hyper_parameters):
    impurity_measure, prune = hyper_parameters
    return f"p1.tree-{impurity_measure}{'-prune' if prune else ''}", P1DecisionTreeClassifier(impurity_measure=impurity_measure, prune=prune)