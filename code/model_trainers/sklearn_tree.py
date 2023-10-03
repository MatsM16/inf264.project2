from sklearn.tree import DecisionTreeClassifier
from model_trainers.trainer import create_train_validate_model_group
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

    plot_sklearn_tree(models)

    return models

def plot_sklearn_tree(group):
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
    ax.set_title("Decisiontree classifier accuracy")
    ax.set_xticks(label_locations + width, labels)
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 110)

    plt.savefig(get_dump_file("sklearn.tree.accuracy.png"))

def create_sklearn_tree(hyper_params):
    (criterion, split) = hyper_params
    return f"sklearn.tree-{criterion}-split_{split}", DecisionTreeClassifier(criterion=criterion, splitter=split)