from src.decisiontree import build_decision_tree
from model_trainers.trainer import create_train_validate_model_group
from plot import plot_accuracy_group

def create_personal_tree(hyper_params):
    (criterion, split) = hyper_params
    return f"personal.tree-{criterion}-{split}", build_decision_tree(criterion=criterion, splitter=split, )