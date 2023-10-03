from sklearn.tree import DecisionTreeClassifier
from model_trainers.trainer import create_train_validate_model_group

def train_sklearn_tree(X, y):
    hyper_parameters = [
        ("gini", "best"), ("entropy", "best"), ("log_loss", "best"),
        ("gini", "random"), ("entropy", "random"), ("log_loss", "random")
        ]

    models = create_train_validate_model_group(
        "sklearn.knn",
        hyper_parameters,
        create_sklearn_tree, 
        X, y)

    models.print_details()

    return models

def create_sklearn_tree(hyper_params):
    (criterion, split) = hyper_params
    return f"sklearn.tree-{criterion}-split_{split}", DecisionTreeClassifier(criterion=criterion, splitter=split)