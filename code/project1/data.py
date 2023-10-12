from random import Random
from statistics import mode
import csv

def split_data(data, *portions:float, shuffle_seed:int = 0):
    """
    Splits and shuffles the data into weighted portions.

    shuffle_index=-1 will case data to not be shuffled.

    Example:
    train, val, test = split_data(data, 70, 15, 15)
    Here train will contain 70% of the data and val/test will have 15% each.
    """
    data = list(data)

    # We do not shuffle if seed is -1.
    # This is because shuffled data is not always wanted.
    if shuffle_seed != -1:
        random = Random(shuffle_seed)
        random.shuffle(data)

    portionsSum = sum(portions)
    portionLists = []
    index = 0
    for portion in portions:
        size = int(len(data) * portion / portionsSum)
        portionLists.append(data[index:index+size])
        index += size

    # Due to integer rounding, the last datapoint might not be included.
    if index == len(data) - 1:
        portionLists[-1].append(data[-1])
        
    return tuple(portionLists)

def split_data_label(data, labels, *portions:float, shuffle_seed:int = 0):
    """
    Splits and shuffles the data and labels into weighted portions.
    If data-0 is now data-123, then label-0 is now label-123.

    shuffle_index=-1 will case data to not be shuffled.

    Example:
    data_train, data_test, labels_train, labels_test = split_data(data, labels, 70, 30)
    Here train will contain 70% of the data and test will have 30%.
    """
    data_portions = list(split_data(data, *portions, shuffle_seed=shuffle_seed))
    label_portions = list(split_data(labels, *portions, shuffle_seed=shuffle_seed))

    return tuple([*data_portions, *label_portions])

def extract_feature(data, feature_index):
    """
    Extract a feature from a set of data-points.
    """
    return [data_point[feature_index] for data_point in data]


def load_csv_dataset(file_path:str, types:list[any]=None, label_index=-1, first_row_is_header=True):
    """
    Loads a dataset from a CSV file.

    file_path: Path to the csv dataset file
    types: Array of constructors (or None) for types in columns.
    label_index: zero-based column-index of the label-column.
    first_row_is_header: Defines if the first row contains column-titles and should be skipped.
    """
    data = []
    labels = []

    with open(file_path, "r") as dataset:
        reader = csv.reader(dataset)

        if first_row_is_header:
            reader.__next__()

        for row in reader:

            if types is not None:
                row = [types[i](row[i]) for i in range(len(row))]

            # Read and remove label from row
            labels.append(row[label_index])
            del row[label_index]

            # Read row as data
            data.append(row)

    return data, labels