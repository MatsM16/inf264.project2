from sklearn.neural_network import MLPClassifier
from model_trainers.trainer import create_train_validate_model_group
from plot import plot_accuracy_group

def train_sklearn_mlp(X, y):
    hyper_parameters = [
        (100,), # Default hidden layers
        (16, ), # Size of image
        (16, 16), # 2x size of image
        (32,),
        (64,),
        (32,32),
        (64,64),
        (100, 100),
        (100, 100, 100)
    ]

    models = create_train_validate_model_group(
        "sklearn.mlp",
        hyper_parameters,
        create_sklearn_mlp, 
        X, y)

    models.print_details()

    plot_accuracy_group(models)

    return models

def create_sklearn_mlp(hyper_parameters):
    return f"sklearn.mlp-{'-'.join([str(param) for param in hyper_parameters])}", MLPClassifier(hidden_layer_sizes=hyper_parameters)