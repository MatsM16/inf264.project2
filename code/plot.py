from model_trainers.trainer import ClassifierGroup, Classifier
from dump import get_dump_file
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_group(group:ClassifierGroup):
    plot_accuracy_models(models=group.models, title=group.name, skip_name_prefix=len(group.name) + 1)

def plot_accuracy_models(models:list[Classifier], title="model", skip_name_prefix=0):
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