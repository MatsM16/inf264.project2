from typing import TypeVar, Sequence, Callable, Generic
from sklearn.model_selection import train_test_split
from time import perf_counter_ns
from logger import log_debug, log
from formats import format_duration

THyperParam = TypeVar('THyperParam')
"""
Type of hyper-parameter.
"""

TModel = TypeVar('TModel')
"""
Type of model.
"""

detailed_logs = True

class ClassifierTestReport:
    """
    Represents the test-performance of a classifier on a given dataset.
    """

    def __init__(self, set_name:str, predictions:int, correct_predictions:int, duration:float):
        """
        Creates a new classifier test report.

        set_name: Name of set the test was performed on.
        predictions: Number of predictions made during test.
        correct_predictions: Number of correct predictions during test.
        duration: Duration (ns) of test.
        """

        self.set_name = set_name
        """
        Name of set the test was performed on.
        """

        self.predictions = predictions
        """
        Number of predictions made during test.
        """

        self.correct_predictions = correct_predictions
        """
        Number of correct predictions made during test.
        """

        self.duration = duration
        """
        Duration (ns) of test.
        """

        self.accuracy =  correct_predictions / predictions
        """
        Accuracy of model in test set. (between 0 and 1)
        """

        self.time_per_prediction = duration / predictions
        """
        Time in ns it takes to make a prediction.
        """

    def print_details_short(self):
        log(f"{self.set_name}: Accuracy={self.accuracy:.0%}, TPP={format_duration(self.time_per_prediction)}")

    def print_details(self):
        log(f"{self.set_name}: Accuracy={self.accuracy:.0%}, TPP={format_duration(self.time_per_prediction)}, Size={self.predictions}, Duration={format_duration(self.duration)}")

class Classifier(Generic[TModel]):
    """
    Represents the performance of a classifier model.
    """

    def __init__(self, model:TModel, name:str, train_size:int, train_time:float):
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

        self.reports:dict[str, ClassifierTestReport] = dict()
        """
        All test reports: {set_name: report}
        """

    def print_details(self):
        log(f"=== Model:\t {self.name}")
        log(f"Training size:\t {self.train_size} pts.")
        log(f"Training time:\t {format_duration(self.train_time)}")
        for report in self.reports.values():
            report.print_details()
        log("")

    def print_details_short(self):
        log(f"=== Model:\t {self.name}")
        for report in self.reports.values():
            report.print_details_short()
        log("")

    def measure_performance(self, set_name:str, X:any, y:any) -> ClassifierTestReport:
        log_debug(f"Testing {self.name} on {set_name}")

        # Test model
        test_time_start = perf_counter_ns()

        y_predicted = self.model.predict(X)

        test_time_end = perf_counter_ns()

        # Calculate performance
        correct_predictions = sum(y[i] == y_predicted[i] for i in range(len(y)))
        test_duration = test_time_end - test_time_start

        log_debug(f"Done testing {self.name} on {set_name} ({(correct_predictions / len(y)):.0%} accurate, Took {format_duration(test_duration)})")

        report = ClassifierTestReport(
            set_name=set_name, 
            predictions=len(y), 
            correct_predictions=correct_predictions,
            duration=test_duration)
            
        self.reports[set_name] = report

        return report

class ClassifierGroup(Generic[TModel, THyperParam]):
    """
    A collection of trained models and their performance.
    """

    def __init__(self, name:str, model_factory:Callable[[THyperParam], TModel]):
        self.name = name
        """
        Name of model group.
        """

        self.models:list[Classifier] = list()
        """
        All models in collection.
        """

        self.model_factory = model_factory
        """
        Factory function that creates a new untrained model from hyper-parameters.
        """

        self.best_model: Classifier = None
        """
        The model that performed best when the group was validated.
        """

        self.best_report: ClassifierTestReport = None
        """
        The report from the best_model.
        """
        
    def __str__(self):
        return self.name

    def print_details(self):
        log(f"====== Group: {self.name}")
        log(f"Best model: {self.best_model.name}\n")
        for model in self.models:
            model.print_details()

    def print_details_short(self):
        log(f"====== Group: {self.name}")
        log(f"Best model: {self.best_model.name}\n")
        for model in self.models:
            model.print_details_short()

    def create_model(self, hyper_parameters:THyperParam, X_train:any, y_train:any) -> Classifier:
        """
        Creates and trains a model with the given hyper-parameters and training set.

        Note: This does not add the model to the group.
        """
        name, model = self.model_factory(hyper_parameters)
        
        log_debug(f"Training {name}")

        train_time_start = perf_counter_ns()

        model.fit(X_train, y_train)
        
        train_time_end = perf_counter_ns()

        train_duration = train_time_end - train_time_start

        log_debug(f"Done training {name} (Took {format_duration(train_duration)})")

        classifier = Classifier(model, name, len(y_train), train_duration)

        # This line measures model perfomance on training data.
        # Can be commented out for faster runtimes if info is not needed.
        classifier.measure_performance("train", X_train, y_train)

        return classifier

    def create_add_model(self, hyper_parameters:THyperParam, X_train:any, y_train:any) -> Classifier:
        """
        Creates and trains a model with the given hyper-parameters and training set.

        Note: The model is added to the group.
        """
        classifier = self.create_model(hyper_parameters, X_train, y_train)
        self.models.append(classifier)
        return classifier

    def validate(self, X_val:any, y_val:any):
        """
        Validates the group with the given dataset.

        Once this function has completed:
        - self.best_model will be set.
        - self.best_report will be set.
        - All models in group will have a report called 'validate'.
        """
        log_debug(f"Validating group {self.name}")

        (self.best_model, self.best_report) = measure_and_find_best(self.models, "validate", X_val, y_val)

def create_train_validate_model_group(group_name:str, hyper_parameters: Sequence[THyperParam], model_factory:Callable[[THyperParam], TModel], X:any, y:any) -> ClassifierGroup[TModel, THyperParam]:
    """
    Creates, trains and measures performance on a group of models.

    group_name: Name of the model group.
    hyper_parameters: List of hyper-parameters.
    model_factory: Creates a model from hyper-parameters: hyper_parameter -> (model_name, model)
    X: Training and validation data-points.
    Y: Training and validation labels.
    """

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.15)

    group = ClassifierGroup(group_name, model_factory)

    def create_train_validate_model(hyper_parameter):
        model = group.create_add_model(hyper_parameter, X_train, y_train)
        return model

    log_debug(f"Creating group {group_name}")

    models = list(map(create_train_validate_model, hyper_parameters))

    group.validate(X_val, y_val)

    log_debug(f"Completed group {group_name}")

    return group

def measure_and_find_best(models:list[Classifier], set_name:str, X:any, y:any) -> tuple[Classifier, ClassifierTestReport]:
    """
    Measures performance on given datset for all models, then returns the best model.
    """

    for model in models:
        model.measure_performance(set_name, X, y)

    return find_best(models, set_name)

def find_best(models:list[Classifier], set_name:str) -> tuple[Classifier, ClassifierTestReport]:
    """
    Assumes models have all been measured with the given dataset.  
    Returns the best model.
    """
    best_model = None
    best_report = None
    best_accuracy = -1
    for model in models:
        report = model.reports[set_name]
        if best_accuracy < report.accuracy:
            best_model = model
            best_report = report
            best_accuracy = report.accuracy
    return best_model, best_report