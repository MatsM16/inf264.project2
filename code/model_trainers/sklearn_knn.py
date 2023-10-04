from sklearn.neighbors import KNeighborsClassifier
from model_trainers.trainer import create_train_validate_model_group
from plot import plot_accuracy_group
from dump import get_dump_file
import matplotlib.pyplot as plt

def train_sklearn_knn(X, y):
    hyper_parameters = [1, 3, 5, 7, 11, 17, 19, 23]

    models = create_train_validate_model_group(
        "sklearn.knn",
        hyper_parameters,
        create_sklearn_knn, 
        X, y)

    models.print_details()

    plot_accuracy_group(models)

    return models

def create_sklearn_knn(k):
    return f"sklearn.knn-k{k}", KNeighborsClassifier(n_neighbors=k)