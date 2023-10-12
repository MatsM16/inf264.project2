from decisiontree import learn, predict

class P1DecisionTreeClassifier:
    def __init__(self, impurity_measure:str, prune=False, prune_portion=.3):
        self.impurity_measure = impurity_measure
        self.prune = prune
        self.prune_portion = prune_portion
        self.tree = None

    def fit(self, X, y):
        self.tree = learn(X, y, self.impurity_measure, self.prune, self.prune_portion)

    def predict(self, X):
        return [predict(x, self.tree) for x in X]