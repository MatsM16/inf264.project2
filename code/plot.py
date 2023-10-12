from model_trainers.trainer import ClassifierGroup, Classifier
from dump import get_dump_file
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def plot_accuracy_group(group:ClassifierGroup):
    plot_accuracy_models(models=group.models, title=group.name, skip_name_prefix=len(group.name) + 1)

def plot_accuracy_models(models:list[Classifier], title="model", skip_name_prefix=0):
    """
    Plots the accuraty of the given models in all the validated sets.
    """
    width = 0.25
    offset = 0

    labels = [model.name[skip_name_prefix:] for model in models]
    label_locations = np.arange(len(labels))

    accuracy = {
        "Training": [int(model.reports["train"].accuracy * 100) for model in models],
        "Validate": [int(model.reports["validate"].accuracy * 100) for model in models]
    }

    if all("test" in model.reports for model in models):
        accuracy["Test"] = [int(model.reports["test"].accuracy * 100) for model in models]

    if all("estimate" in model.reports for model in models):
        accuracy["Estimate"] = [int(model.reports["estimate"].accuracy * 100) for model in models]

    fig, ax = plt.subplots(layout="constrained")
    for label, accuracy_data in accuracy.items():
        bars = ax.bar(label_locations + offset, accuracy_data, width, label=label)
        ax.bar_label(bars, padding=3)
        offset += width

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title} accuracy")
    ax.set_xticks(label_locations + width, labels)
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 110)

    plt.savefig(get_dump_file(f"{title}.accuracy.png"))
    plt.close(fig='all')

def plot_confusion_matrix(classifier:Classifier, X, y):
    """
    Plots the confusion matrix for the classifier on the given dataset.
    """
    unique_y = list(set(y))
    size = len(unique_y)
    predictions = classifier.model.predict(X)
    matrix = [0] * size * size

    for i in range(len(y)):
        if predictions[i] == y[i]: continue
        column = unique_y.index(y[i])
        row = unique_y.index(predictions[i])
        matrix[column + row * size] += 1

    vmax = max(matrix)
    matrix = np.array(matrix).reshape(size, size)

    formatter = FuncFormatter(lambda x, pos: f"{int(x):x}")
    plt.imshow(matrix, vmin=0, vmax=vmax, cmap="gray")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.title(f"Confusion matrix for {classifier.name}")
    plt.xlabel("Label")
    plt.ylabel("Prediction")
    plt.xticks(ticks=[float(i) for i in range(16)])
    plt.yticks(ticks=[float(i) for i in range(16)])
    plt.xlim(0, 16)
    plt.ylim(0, 16)
    plt.savefig(get_dump_file(f"{classifier.name}.confusion.png"))
    plt.close(fig='all')
