import numpy as np
import platform
import multiprocessing
import psutil
from logger import log
from dump import start_timestamp
from formats import format_bytes, format_duration
from sklearn.model_selection import train_test_split
from time import perf_counter_ns

# Import model trainers
from model_trainers.sklearn_knn import train_sklearn_knn
from model_trainers.sklearn_tree import train_sklearn_tree
from model_trainers.sklearn_svm import train_sklearn_svm

log(f"====== Starting program")
log(f"Logs are writted to 'dump/'-folder. Look for the timestamp.")
log(f"Timestamp:{start_timestamp}")
log("")
log(f"=== Machine information")
log(f"Performance will vary from machine to machine. Take this into consideration.")
log(f"OS: {platform.platform()}")
log(f"CPU: {platform.processor()}, {multiprocessing.cpu_count()} cores")
log(f"MEMORY: {format_bytes(psutil.virtual_memory().total)}")
log("")
log(f"====== ====== ====== ====== ====== ======")
log("\n")

time_start = perf_counter_ns()

# Load datset
X = np.load("emnist_hex_images.npy")
y = np.load("emnist_hex_labels.npy")

# Split data into subsets
# X_train is used to train and find the best version of each model.
# X_val is used to find the best model
# X_test is used to estimate performance on unseen data.
# It is very important that the model trainers never see the X_val or X_test datasets.
# The model trainers are free to use the train-dataset however they please.
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15)

best_sklearn_knn = train_sklearn_knn(X_train, y_train)

#best_sklearn_tree = train_sklearn_tree(X_train, y_train)

#best_sklearn_svm = train_sklearn_svm(X_train, y_train)

time_end = perf_counter_ns()
log(f"Program completed after {format_duration(time_end - time_start)}")