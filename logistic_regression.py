import random
import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=100000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None

    def fit(self, X, y, reg_coef=0.01):
        X = np.c_[X, np.ones(X.shape[0])]
        n_samples, n_features = X.shape
        print(n_samples, n_features)
        self.weights = np.zeros(n_features)

        for _ in range(self.n_iters):
            pos = random.randint(0, n_samples - 1)
            linear_model = np.dot(X[pos], self.weights)
            y_predicted: float = self._sigmoid(linear_model)

            dw = (X[pos].T * (y_predicted - y[pos])) + reg_coef * self.weights
            self.weights -= self.lr * dw

    def predict(self, x):
        x = np.append(x, 1)
        linear_model = np.dot(x, self.weights)
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = 1 if y_predicted >= 0.5 else 0
        return y_predicted_cls

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def score(self, x, y):
        correct = 0
        for i in range(len(x)):
            if self.predict(x[i]) == y[i]:
                correct += 1
        return correct / len(x)
