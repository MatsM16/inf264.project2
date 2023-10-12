from decisiontree import learn, predict
from data import split_data_label, load_csv_dataset
from measure import measure_performance
from sklearn import tree

# Loads the wine dataset
# The file path will change depending on the current working directory.
data, labels = load_csv_dataset(
    "wine_dataset.csv", types=[float, float, float, float, float, int]
)

test_size = 15

X_train_val, X_test, y_train_val, y_test = split_data_label(
    data, labels, 100 - test_size, test_size
)

reports = []


def train_measure_and_report(train_size, validate_size):
    X_train, X_val, y_train, y_val = split_data_label(
        X_train_val, y_train_val, train_size, validate_size
    )

    print(f"Dataset:\twine_dataset.csv")
    print(f"Size:\t\t{len(data)} points")
    print(f"Training:\t{len(X_train)} points ({(len(X_train)/len(data)):.0%})")
    print(f"Validate:\t{len(X_val)} points ({(len(X_val)/len(data)):.0%})")
    print(f"Test:\t\t{len(X_test)} points ({(len(X_test)/len(data)):.0%})\n")

    config_name = f".train{train_size}-val{validate_size}-test{test_size}"

    # Learn dataset using various models
    entropy = learn(X_train, y_train, impurity_measure="entropy")
    gini = learn(X_train, y_train, impurity_measure="gini")
    entropy_pruned = learn(X_train, y_train, impurity_measure="entropy", prune=True)
    gini_pruned = learn(X_train, y_train, impurity_measure="gini", prune=True)

    # Learn dataset using various sklearn models
    sklearn_entropy = tree.DecisionTreeClassifier(criterion="entropy")
    sklearn_entropy.fit(X_train, y_train)
    sklearn_gini = tree.DecisionTreeClassifier(criterion="gini")
    sklearn_gini.fit(X_train, y_train)

    # Measure performance
    # Measurements are added to a reports-list which we will use later for picking the best model.
    reports.append(measure_performance(entropy, "entropy" + config_name, X_val, y_val))
    reports.append(measure_performance(gini, "gini" + config_name, X_val, y_val))
    reports.append(
        measure_performance(
            entropy_pruned, "entropy.pruned" + config_name, X_val, y_val
        )
    )
    reports.append(
        measure_performance(gini_pruned, "gini.pruned" + config_name, X_val, y_val)
    )
    reports.append(
        measure_performance(
            sklearn_entropy, "sklearn.entropy" + config_name, X_val, y_val
        )
    )
    reports.append(
        measure_performance(sklearn_gini, "sklearn.gini" + config_name, X_val, y_val)
    )

    # Some padding to help readbility in the output
    print("--- --- --- --- --- --- ---")
    print("")


# Train and measure with many different train/val splits
train_measure_and_report(train_size=15, validate_size=70)
train_measure_and_report(train_size=45, validate_size=40)
train_measure_and_report(train_size=70, validate_size=15)

# Find the most accurate model
most_accurate = None
for report in reports:
    if most_accurate is None or most_accurate.accuracy < report.accuracy:
        most_accurate = report

print(f"Most accurate model: {most_accurate.name}")
measure_performance(most_accurate.model, "best on test_set", X_test, y_test)

# Find most accurate model that we made
most_accurate = None
for report in reports:
    if (
        most_accurate is None
        or most_accurate.accuracy < report.accuracy
        and report.name.startswith("sklearn") is False
    ):
        most_accurate = report

print(f"Most accurate model from us: {most_accurate.name}")
measure_performance(most_accurate.model, "best_from_us on test_set", X_test, y_test)
