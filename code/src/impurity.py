from sys import float_info
from math import log2


def entropy(labels: list[any]):
    """
    Calculates the entropy in given set of labels.
    """
    entropy = 0
    unique = set(labels)
    for value in unique:
        probability = labels.count(value) / len(labels)
        entropy += -probability * log2(probability)
    return entropy


def gini(labels: list[any]):
    """
    Calculates the entropy in given set of labels.
    """
    entropy = 1
    unique = set(labels)
    for value in unique:
        probability = labels.count(value) / len(labels)
        entropy -= probability**2
    return entropy
