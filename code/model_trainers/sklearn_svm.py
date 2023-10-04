from sklearn.svm import SVC
from model_trainers.trainer import create_train_validate_model_group
from plot import plot_accuracy_group
from dump import get_dump_file
import matplotlib.pyplot as plt
import numpy as np

def train_sklearn_svm(X, y):
    hyper_parameters = [
        ("poly", 1),
        ("poly", 2),
        ("poly", 3),
        ("poly", 4),
        ("poly", 5),
        ("poly", 6),
        ("rbf", None),
        ("sigmoid", None)
    ]

    # The linear kernel is not included due to being extremely slow.
    # After training for 2+ hours, it was still not completed,
    # so we decided to exclude it.

    models = create_train_validate_model_group(
        "sklearn.svm",
        hyper_parameters,
        create_sklearn_svm, 
        X, y)

    models.print_details()

    plot_accuracy_group(models)

    return models

def create_sklearn_svm(hyper_parameters):
    (kernel, degree) = hyper_parameters
    if degree is None:
        return f"sklearn.svm-{kernel}", SVC(kernel=kernel)
    else:
        return f"sklearn.svm-{kernel}{degree}", SVC(kernel=kernel, degree=degree)