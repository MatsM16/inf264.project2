from sklearn.neighbors import KNeighborsClassifier
from model_trainers.trainer import create_train_measure_classifiers

def train_sklearn_knn(X, y):
    # hyper_parameters = [1, 3, 5, 11, 17, 31]
    hyper_parameters = [1, 3]

    models = create_train_measure_classifiers(
        "sklearn.knn",
        hyper_parameters,
        create_sklearn_knn, 
        X, y)

    models.print_details()

    return models.best_model.model

def create_sklearn_knn(k):
    return f"sklearn.knn-k{k}", KNeighborsClassifier(n_neighbors=k)