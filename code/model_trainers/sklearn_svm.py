from sklearn.svm import SVC
from model_trainers.trainer import create_train_validate_model_group
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

    plot_sklearn_svm(models)

    return models


def plot_sklearn_svm(group):
    width = 0.25
    offset = 0

    labels = [model.name[len(group.name) + 1:] for model in group.models]
    label_locations = np.arange(len(labels))


    accuracy = {
        "Training": [int(model.reports["train"].accuracy * 100) for model in group.models],
        "Validate": [int(model.reports["validate"].accuracy * 100) for model in group.models]
    }

    fig, ax = plt.subplots(layout="constrained")
    for label, accuracy_data in accuracy.items():
        bars = ax.bar(label_locations + offset, accuracy_data, width, label=label)
        ax.bar_label(bars, padding=3)
        offset += width

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SVM classifier accuracy")
    ax.set_xticks(label_locations + width, labels)
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 110)

    plt.savefig(get_dump_file("sklearn.svm.accuracy.png"))


def create_sklearn_svm(hyper_parameters):
    (kernel, degree) = hyper_parameters
    if degree is None:
        return f"sklearn.svm-{kernel}", SVC(kernel=kernel)
    else:
        return f"sklearn.svm-{kernel}{degree}", SVC(kernel=kernel, degree=degree)