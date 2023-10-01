from typing import TypeVar, Sequence, Callable, Generic
from sklearn.model_selection import train_test_split
from time import perf_counter_ns
from parallel import parallel_map

THyperParam = TypeVar('THyperParam')
"""
Type of hyper-parameter.
"""

TModel = TypeVar('TModel')
"""
Type of model.
"""

class Classifier(Generic[TModel]):
    """
    Represents the performance of a classifier model.
    """

    def __init__(self, model:TModel, name:str, train_size:int, train_time:float, test_size:int, test_time:float, test_correct:int):
        """
        Creates a new classifier training result.

        model: The model that was trained.
        name: Display name to help identify the model.
        train_size: Size of training set.
        train_time: Duration (ns) of training.
        test_size: Size of testing set.
        test_time: Duration (ns) of testing.
        test_correct: Number of correctly labelled test-datapoints.
        """
        self.model = model
        """
        Model that was measured.
        """

        self.name = name
        """
        Display-name of model
        """

        self.train_size = train_size
        """
        Size of training set.
        """

        self.train_time = train_time
        """
        Duration (ns) of training.
        """

        self.test_size = test_size
        """
        Size of testing set.
        """

        self.test_time = test_time
        """
        Duration (ns) of testing.
        """

        self.test_correct = test_correct
        """
        Number of correctly labelled test-data.
        """

        self.accuracy = test_correct / test_size
        """
        Accuracy of the model on test data.
        """

        self.predict_time = test_time / test_size
        """
        Time (ns) per prediction.
        """

    def print_details(self):
        print(f"=== Model:\t {self.name}")
        print(f"Training size:\t {self.train_size} pts.")
        print(f"Training time:\t {format_duration_ns(self.train_time)}")
        print(f"Test size:\t {self.test_size} pts.")
        print(f"Test time:\t {format_duration_ns(self.test_time)}")
        print(f"Accuracy:\t {self.accuracy:.1%}")
        print(f"Time pr predict: {format_duration_ns(self.predict_time)}")
        print()

    def print_details_short(self):
        print(f"=== Model:\t {self.name}")
        print(f"Accuracy:\t {self.accuracy:.1%}")
        print(f"Time pr predict: {format_duration_ns(self.predict_time)}")
        print()

class ClassifierGroup(Generic[TModel, THyperParam]):
    """
    A collection of trained models and their performance.
    """

    def __init__(self, name:str, models:list[Classifier[TModel]], best_model:Classifier[TModel], model_factory:Callable[[THyperParam], TModel]):
        self.name = name
        """
        Name of model group.
        """

        self.models = models
        """
        All models in collection.
        """

        self.best_model = best_model
        """
        The most accurate model.
        """

        self.model_factory = model_factory
        """
        Factory function that creates a new untrained model from hyper-parameters.
        """
        
    def __str__(self):
        return self.name

    def print_details(self):
        print(f"====== Group: {self.name}")
        print(f"Best model: {self.best_model.name}\n")
        for model in self.models:
            model.print_details()

    def print_details_short(self):
        print(f"====== Group: {self.name}")
        print(f"Best model: {self.best_model.name}\n")
        for model in self.models:
            model.print_details_short()

def create_train_measure_classifiers(group_name:str, hyper_parameters: Sequence[THyperParam], model_factory:Callable[[THyperParam], TModel], X:any, y:any) -> ClassifierGroup[TModel, THyperParam]:
    """
    Trains and measures a model-type with various provided hyper-parameters.

    group_name: Name of the model group.
    hyper_parameters: List of hyper-parameters.
    model_factory: Creates a model from hyper-parameters: hyper_parameter -> (model_name, model)
    X: Training and validation data-points.
    Y: Training and validation labels.
    """

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.15)

    def model_from_parameters(hyper_parameter):
        name, model = model_factory(hyper_parameter)
        return train_measure_classifier(name, model, hyper_parameter, X_train, y_train, X_val, y_val)

    models = parallel_map(model_from_parameters, hyper_parameters)

    best_model = max(models, key=lambda m : m.accuracy)

    return ClassifierGroup(group_name, models, best_model, model_factory)

def train_measure_classifier(name:str, model:TModel, hyper_parameters:THyperParam, X_train:any, y_train:any, X_test:any, y_test:any) -> Classifier[TModel]: 
    """
    Trains and measures performance of a model.

    name: Display name of model.
    model: Instance of model to train and measure.
    hyper_parameters: Hyper-parameters model was created with.
    X_train: Training dataset.
    y_train: Training labels.
    X_test: Testing dataset.
    y_test: Testing labels.
    """
    # Train model
    train_time_start = perf_counter_ns()

    model.fit(X_train, y_train)
    
    train_time_end = perf_counter_ns()

    # Test model
    test_time_start = perf_counter_ns()

    y_test_predicted = model.predict(X_test)

    test_time_end = perf_counter_ns()
    
    # Calculate performance
    test_correct = 0
    for i in range(len(y_test)):
        if y_test_predicted[i] == y_test[i]:
            test_correct += 1 

    return Classifier(model, name,
        train_size=len(X_train),
        train_time=train_time_end - train_time_start,
        test_size=len(X_test),
        test_time=test_time_end - test_time_start,
        test_correct=test_correct)

def format_duration_ns(duration:float) -> str:
    """
    Formats a duration measurement to a human readable format.
    """
    if duration < 1_000_000:
        # I dont really care about decimal places for nanoseconds.
        return f"{duration:.0f}ns"
    
    duration /= 1_000_000
    if duration < 1_000:
        return f"{duration:.2f}ms"

    duration /= 1_000
    return f"{duration:.2f}s"