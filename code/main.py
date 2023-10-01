import numpy as np
from sklearn.model_selection import train_test_split

# Import model trainers
from model_trainers.sklearn_knn import train_sklearn_knn
from model_trainers.sklearn_tree import train_sklearn_tree

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

best_sklearn_tree = train_sklearn_tree(X_train, y_train)