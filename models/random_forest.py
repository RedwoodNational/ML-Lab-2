import numpy as np
from collections import Counter

from .decision_tree import DTClassifier

class RFClassifier():
    def __init__(self, max_depth=5, n_estimators=100):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.forest = [None] * n_estimators

    def fit(self, X, y):
        for i in range(self.n_estimators):
            
            # bootstrap X and y
            idx = np.random.choice(X.shape[0], X.shape[0])
            X, y = X[tuple([idx])], y[tuple([idx])]
            
            # max features - sqrt(n) for classification
            n_features = np.sqrt(X.shape[1]).astype(int)
            features = np.random.choice(X.shape[1], n_features, replace=False)
            X = X[:, features]
            
            self.forest[i] = DTClassifier(self.max_depth)
            self.forest[i].fit(X, y)

    def predict(self, X):
        tree_labels = np.zeros(X.shape[0])
        preds = np.zeros((self.n_estimators, X.shape[0]))
        for i in range(self.n_estimators):
            preds[i] = self.forest[i].predict(X)
        for i in range(len(tree_labels)):
            tree_labels[i] = Counter(preds[:, i]).most_common(1)[0][0]
        return tree_labels.astype(int)