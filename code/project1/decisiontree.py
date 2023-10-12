from statistics import mode
from data import extract_feature, split_data_label
import impurity


class DecisionTreeLeaf:
    """
    Represents the end of a decision tree and contains a single value.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.formatIndented("")

    def __len__(self):
        return 1

    def formatIndented(self, indent):
        return f"Then {self.value}"

class DecisionTreeContinuous:
    """
    Represents a decision on continuous values.
    Contains a tree for included and a tree for excluded values.
    """

    def __init__(self, feature_index, split_value):
        self.split_value = split_value
        self.feature_index = feature_index
        self.overTree = None
        self.underTree = None

    def __str__(self):
        return self.formatIndented("  ")
    
    def __len__(self):
        return 1 + len(self.overTree) + len(self.underTree)

    def formatIndented(self, indent):
        formatted_over = self.overTree.formatIndented(indent + "  ")
        formatted_under = self.underTree.formatIndented(indent + "  ")
        return f"Is feature_{self.feature_index} > {self.split_value}\n{indent}Yes: {formatted_over}\n{indent}No: {formatted_under}"

    def get_subtree(self, data_point):
        is_over = data_point[self.feature_index] > self.split_value
        return self.overTree if is_over else self.underTree

    def split(self, data, labels):
        over_data = []
        over_labels = []
        under_data = []
        under_labels = []

        for i in range(len(data)):
            # If data belongs to over-tree
            if data[i][self.feature_index] > self.split_value:
                # Add data to over-set
                over_data.append(data[i])
                over_labels.append(labels[i])
            # Otherwise
            else:
                # Add data to under-set
                under_data.append(data[i])
                under_labels.append(labels[i])

        return over_data, over_labels, under_data, under_labels

def prune_tree(tree, prune_data, prune_labels):
    leaves = prune_find_leaves(tree)

    for i in range(100):
        # Make a copy of the leaf list that is safe to iterate
        # when we modify the original list
        safe_iteration_copy = list(leaves)
        for leaf in safe_iteration_copy:
            parent = leaf.parent

            if hasattr(parent, "parent") is False:
                # We cannot prune the root node
                continue

            grand_parent = parent.parent
            is_over = grand_parent.overTree == parent

            accuracy_before = prune_accuracy(tree, prune_data, prune_labels)

            if is_over:
                grand_parent.overTree = leaf
            else:
                grand_parent.underTree = leaf
            leaf.parent = grand_parent

            accuracy_after = prune_accuracy(tree, prune_data, prune_labels)

            if accuracy_after <= accuracy_before:
                if is_over:
                    grand_parent.overTree = parent
                else:
                    grand_parent.underTree = parent
                leaf.parent = parent

                # Remove leaf because it cannot be pruned
                leaves.remove(leaf)

    return tree

def prune_accuracy(tree, prune_data, prune_labels):
    correct = 0

    for i in range(len(prune_data)):
        predict_label = predict(prune_data[i], tree)
        correct_label = prune_labels[i]
        if predict_label == correct_label:
            correct += 1

    return correct / len(prune_data)

def prune_find_leaves(tree, leaves=[]):
    if isinstance(tree, DecisionTreeLeaf):
        leaves.append(tree)
        return leaves
    else:
        prune_find_leaves(tree.overTree, leaves)
        prune_find_leaves(tree.underTree, leaves)
        return leaves

def build_decision_tree(
    data: list[list[float]], labels: list[float], impurity_measure: str
):
    """
    Learns a data-set using a greedy decision-tree algorithm.
    """

    if len(data) != len(labels):
        raise Exception("Number of data must match the number of labels.")

    if len(data) < 1:
        # Should never happen
        return DecisionTreeLeaf(None)

    feature_count = len(data[0])

    # If all labels are the same
    if all(label == labels[0] for label in labels):
        # Return leaf node with that label
        return DecisionTreeLeaf(labels[0])

    # If every feature is the same in every data-point
    if all(
        all(data[i][feature] == data[0][feature] for i in range(len(data)))
        for feature in range(feature_count)
    ):
        # Return leaf node with most common label
        most_common_label = mode(labels)
        return DecisionTreeLeaf(most_common_label)

    best_feature_index = -1
    best_feature_values = None

    if impurity_measure == "entropy":
        # Find parent entropy
        entropy = impurity.entropy(labels)

        best_information_gain = None

        # Find feature maximising information gain
        for feature_index in range(feature_count):
            feature_values = extract_feature(data, feature_index)

            # Sum entropy of values given feature
            feature_entropy = 0
            for unique_value in set(feature_values):
                labels_given_value = [
                    labels[i]
                    for i in range(len(data))
                    if data[i][feature_index] == unique_value
                ]
                feature_entropy += (
                    impurity.entropy(labels_given_value)
                    * feature_values.count(unique_value)
                    / len(feature_values)
                )

            # Information gain is original entropy minus entropy given feature values
            information_gain = entropy - feature_entropy

            # Keep track of the best feature
            if (
                best_information_gain is None
                or best_information_gain < information_gain
            ):
                best_feature_index = feature_index
                best_feature_values = feature_values
                best_information_gain = information_gain

    elif impurity_measure == "gini":
        # Find parent entropy
        best_gini = None

        # Find feature maximising information gain
        for feature_index in range(feature_count):
            feature_values = extract_feature(data, feature_index)

            feature_gini = impurity.gini(feature_values)

            # Sum entropy of values given feature
            # feature_entropy = 0
            # for unique_value in set(feature_values):
            # labels_given_value = [labels[i] for i in range(len(data)) if data[i][feature_index] == unique_value]
            # feature_entropy += impurity.entropy(labels_given_value) * feature_values.count(unique_value) / len(feature_values)

            # Keep track of the best feature
            if best_gini is None or best_gini < feature_gini:
                best_feature_index = feature_index
                best_feature_values = feature_values
                best_gini = feature_gini
    else:
        raise Exception("Impurity measure is not supported: " + impurity_measure)

    # We split the feature 50/50.
    # This assumes the feature is continuous.
    # There are probably better ways to calculate this.
    feature_split = (min(best_feature_values) + max(best_feature_values)) / 2

    decision = DecisionTreeContinuous(best_feature_index, feature_split)
    over_data, over_labels, under_data, under_labels = decision.split(data, labels)

    if len(over_data) is 0 or len(under_labels) is 0:
        raise Exception("Something went wrong")

    decision.overTree = build_decision_tree(over_data, over_labels, impurity_measure)
    decision.underTree = build_decision_tree(under_data, under_labels, impurity_measure)
    decision.overTree.parent = decision
    decision.underTree.parent = decision

    # I have yet to experience an uncompleted tree.
    # But, there might be an idea to check if the tree is valid before returning.
    return decision

def learn(
    data: list[list[float]],
    labels: list[float],
    impurity_measure: str,
    prune=False,
    prune_portion: float = 0.3,
):
    """
    Learns a data-set using a greedy decision-tree algorithm.

    Supports pruning the tree after.
    """

    if prune is False:
        return build_decision_tree(data, labels, impurity_measure)

    # We set suffle_seed to -1 so data is not shuffled.
    # This will lead to more consistent trees.
    X_train, X_prune, y_train, y_prune = split_data_label(
        data, labels, (1 - prune_portion), prune_portion, shuffle_seed=-1
    )

    tree = build_decision_tree(X_train, y_train, impurity_measure)

    prune_tree(tree, X_prune, y_prune)

    return tree

def predict(data_point, tree):
    """
    Predicts the label for the data-point using a pre-trained desition tree.
    """

    if isinstance(tree, DecisionTreeLeaf):
        return tree.value

    if isinstance(tree, DecisionTreeContinuous):
        subtree = tree.get_subtree(data_point)
        return predict(data_point, subtree)

    raise Exception("Tree type is not supported")
