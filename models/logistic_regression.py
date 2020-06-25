
import numpy as np

class LogRegression():
    def __init__(self, lr=0.01, iters=500):
        self.lr = lr
        self.iters = iters

    def getB(self, X): #Ax + b
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = self.getB(X)
        self.coefs = np.zeros(X.shape[1])  # weights

        for _ in range(self.iters):
            z = np.dot(X, self.coefs)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.coefs -= self.lr * gradient

    def predict(self, X):
        X = self.getB(X)
        return self.sigmoid(np.dot(X, self.coefs)).round()