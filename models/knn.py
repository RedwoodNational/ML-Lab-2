import numpy as np
from collections import Counter


class KNNClassifier():
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        distances = self.get_distances(X)

        x = distances.shape[0]
        preds = np.zeros(x)

        for i in range(x):
            labels = self.y[np.argsort(distances[i, :])].flatten()
            nn = labels[:self.k]

            c = Counter(nn)
            preds[i] = c.most_common(1)[0][0]

        return preds

    def get_distances(self, X):

        return np.sqrt(-2 * np.dot(X, self.X.T) + np.square(self.X).sum(axis=1) + np.matrix(np.square(X).sum(axis=1)).T)
