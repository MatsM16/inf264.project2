from decisiontree import predict
from time import perf_counter_ns
from sklearn.tree import DecisionTreeClassifier


class ClassifierPerformance:
    """
    Contains a performance measurement of a classifier model.

    model: Model used to make predictions
    name: Name of the model
    predictions: Total number of preditions made during the test
    predictions_correct: Number of predictions that was correct
    duration_ns: Duration of time spent from before first to after last prediction.
    """

    def __init__(self, model, name, predictions, predictions_correct, duration_ns):
        self.model = model
        self.name = name
        self.predictions = predictions
        self.predictions_correct = predictions_correct
        self.predictions_incorrect = predictions - predictions_correct
        self.accuracy = predictions_correct / predictions
        self.duration_ns = duration_ns
        self.duration_predict_ns = duration_ns / predictions
        self.nodes = len(model) if hasattr(model, "__len__") else "Unknown"

    def __str__(self):
        return f"Performance of '{self.name}'\nPredictions:\t{self.predictions}\nCorrect:\t{self.predictions_correct}\nIncorrect:\t{self.predictions_incorrect}\nAccuracy:\t{self.accuracy:.1%}\nTime / predict:\t{int(self.duration_predict_ns)}ns\nNodes:\t{self.nodes}\n"


def measure_performance(tree, name, data, labels):
    """
    Measures performance of tree and prints measurement.
    """
    data_points = len(data)
    correct = 0

    make_prediction = create_predict_function(tree)

    # Get timestamp before predicting
    time_start = perf_counter_ns()

    for i in range(data_points):
        predict_label = make_prediction(data[i])
        correct_label = labels[i]
        if predict_label == correct_label:
            correct += 1

    # Get timestamp after predicting
    time_end = perf_counter_ns()

    performance = ClassifierPerformance(
        tree, name, data_points, correct, time_end - time_start
    )

    print(performance)

    return performance


def create_predict_function(tree):
    """
    Used to unify API for sklearn model and for our own model.
    """
    if isinstance(tree, DecisionTreeClassifier):
        return lambda x: tree.predict([x])
    return lambda x: predict(x, tree)
