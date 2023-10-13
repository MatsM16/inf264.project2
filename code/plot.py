from model_trainers.trainer import ClassifierGroup, Classifier
from dump import get_dump_file
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from formats import format_label

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

    X_error = []
    y_error = []
    y_error_predicted = []

    for i in range(len(y)):
        # Skip correct predictions.
        # They only reduce visibility of interresting errors.
        if y[i] == predictions[i]: continue
        column = unique_y.index(y[i])
        row = unique_y.index(predictions[i])
        matrix[column + row * size] += 1
        X_error.append(X[i])
        y_error.append(y[i])
        y_error_predicted.append(predictions[i])

    vmax = max(matrix)
    matrix = np.array(matrix).reshape(size, size)

    formatter = FuncFormatter(lambda x, pos: format_label(x))
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

    plot_dataexamples(X_error, y_error, y_error_predicted, max_examples=5, title="mislabelled")

def plot_label_distribution(labels, dataset:str):
    """
    Plots the distribution of labels in a given dataset.
    """

    unique_labels = set(labels)
    unique_labels_sorted = sorted(unique_labels)

    label_titles = []
    label_counts = []
    for label in sorted(set(labels)):
        label_titles.append(format_label(label))
        label_counts.append(np.count_nonzero(labels == label))

    plt.title(f"{dataset} label distribution ({len(labels)} datapoints)")
    plt.xlabel("Labels")
    plt.ylabel("Occurances")
    plt.bar(label_titles, label_counts, width=0.4)
    plt.savefig(get_dump_file(f"distribution.{dataset}.png"))
    plt.close(fig='all')

def plot_dataexamples(X, y, y_predicted = None, y_subset = None, max_examples = 4, title=None):
    """
    Plots examples of images and labels. 
    Can also plot predicted labels if provided.

    X: Images.
    y: Labels for images.
    y_predicted: Labels predicted by classifier.
    y_subset: Only show examples from labels in this set.
    max_examples: Maximum number of examples for a given label.
    title: Title describing what the examples are.
    """
    if y_subset is None: y_subset = y
    y_subset = list(sorted(set(y_subset)))

    total_examples = 0
    examples = dict()
    for i in range(len(y)):
        if total_examples > max_examples * len(y_subset): break
        if y[i] not in y_subset: continue
        if y[i] not in examples: examples[y[i]] = list()
        if len(examples[y[i]]) < max_examples:
            examples[y[i]].append((X[i], None if y_predicted is None else y_predicted[i]))
            total_examples += 1

    plt.title(title)
    fig, axs = plt.subplots(len(y_subset), max_examples, figsize=(3*max_examples, 3*len(y_subset)))

    for i in range(len(y_subset)):
        label = y_subset[i]
        label_examples = examples[label]
        for j in range(max_examples):
            datapoint, predicted = (None, None) if len(label_examples) < j else label_examples[j]
            if datapoint is None: continue
            ax = axs[i,j] if max_examples > 1 else axs[i]
            if j == 0: ax.set_title(f"Actual {format_label(label)}")
            if predicted is not None: ax.set_xlabel(f"Predicted {format_label(predicted)}")
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(datapoint.reshape(20, 20), vmin=0, vmax=255, cmap="gray")

    plt.savefig(get_dump_file(f"examples.{'some' if title is None else title}.png"))
    plt.close(fig='all')
        
