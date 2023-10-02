from sklearn.svm import SVC
from model_trainers.trainer import create_train_measure_classifiers

def train_sklearn_svm(X, y):
    hyper_parameters = [
        ("poly", 1),
        ("poly", 2),
        ("poly", 3),
        ("poly", 4),
        ("linear", None),
        ("rbf", None),
        ("sigmoid", None)
    ]

    models = create_train_measure_classifiers(
        "sklearn.knn",
        hyper_parameters,
        create_sklearn_svm, 
        X, y)

    models.print_details()

    return models.best_model.model

def create_sklearn_svm(hyper_parameters):
    (kernel, degree) = hyper_parameters
    if degree is None:
        return f"sklearn.svm-{kernel}", SVC(kernel=kernel)
    else:
        return f"sklearn.svm-{kernel}-{degree}", SVC(kernel=kernel, degree=degree)