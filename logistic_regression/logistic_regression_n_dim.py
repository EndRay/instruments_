import numpy as np

from logistic_regression_main import LogisticRegression


class LogisticRegressionNDim():
    def __init__(self, dimensions, learning_rate=0.01, n_iters=100000):
        self.models = []
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        for i in range(dimensions):
            self.models.append(LogisticRegression(learning_rate, n_iters))

    def train(self, x, y, reg_coef=0.01):
        for i in range(len(self.models)):
            print("Training model for dimension " + str(i) + "...")
            train_data_x = np.array(x)
            train_data_y = np.array([y[j][i] for j in range(len(y))])
            self.models[i].fit(train_data_x, train_data_y, reg_coef)

    def score(self, x, y):
        correct = 0
        for i in range(len(x)):
            predicted = []
            for j in range(len(self.models)):
                predicted.append(self.models[j].predict(x[i]))
            if predicted == y[i]:
                correct += 1
        return correct / len(x)

