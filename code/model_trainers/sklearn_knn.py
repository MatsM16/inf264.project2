from sklearn.neighbors import KNeighborsClassifier
from model_trainers.trainer import create_train_validate_model_group
from dump import get_dump_file
import matplotlib.pyplot as plt

def train_sklearn_knn(X, y):
    hyper_parameters = [1, 3, 5, 11, 17]

    models = create_train_validate_model_group(
        "sklearn.knn",
        hyper_parameters,
        create_sklearn_knn, 
        X, y)

    models.print_details()

    plot_sklearn_knn(models, hyper_parameters)

    return models

def plot_sklearn_knn(group, k):
    accuracy_train_by_k = [int(model.reports["train"].accuracy * 100) for model in group.models]
    accuracy_val_by_k = [int(model.reports["validate"].accuracy * 100) for model in group.models]

    plt.plot(k, accuracy_train_by_k, label="Training set")
    plt.plot(k, accuracy_val_by_k, label="Validation set")
    plt.legend()
    plt.title("Accuracy by k")
    plt.xlabel("Neighbours (K)")
    plt.ylabel("Accuracy (%)")
    plt.savefig(get_dump_file(f"sklearn.knn.accuracy_by_k.png"))

def create_sklearn_knn(k):
    return f"sklearn.knn-k{k}", KNeighborsClassifier(n_neighbors=k)