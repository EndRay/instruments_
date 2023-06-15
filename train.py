import random
import time

from sklearn.neural_network import MLPClassifier

DATA_PERCENTAGE = [0.01, 0.1, 0.2, 0.3, 0.625, 1]


def train(train_data, test_data, solver='adam', hidden_layer_sizes=[400, 400], activation='relu', loss_curve_iterations=0,
          data_percentage_accuracy=False, train=False):
    result = {}
    if loss_curve_iterations > 0:
        print('Drawing loss curve...')
        clf = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes, random_state=1, activation=activation,
                            verbose=True, warm_start=True, max_iter=1)
        loss_curve = []
        for i in range(loss_curve_iterations):
            clf.fit([x[0] for x in train_data], [x[1] for x in train_data])
            loss_curve.append(clf.loss_)
            print('Iteration ' + str(i) + ': ' + str(clf.loss_))
        result['loss_curve'] = loss_curve
    if data_percentage_accuracy:
        print('Calculating accuracy for different data percentages...')
        result['data_percentage_accuracy'] = []
        for data_percentage in DATA_PERCENTAGE:
            start = time.time()
            clf = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes, random_state=1,
                                activation=activation, verbose=True, max_iter=10)
            clf.fit([x[0] for x in train_data[:int(len(train_data) * data_percentage)]],
                    [x[1] for x in train_data[:int(len(train_data) * data_percentage)]])
            end = time.time()
            result['data_percentage_accuracy'].append(clf.score([x[0] for x in test_data], [x[1] for x in test_data]))
            if data_percentage == 1:
                result['time'] = end - start
                result['accuracy'] = result['data_percentage_accuracy'][-1]
            print('Data percentage: ' + str(data_percentage) + ', accuracy: ' + str(
                result['data_percentage_accuracy'][-1]))
    elif train:
        print('Training...')
        clf = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes, random_state=1,
                            activation=activation, verbose=True)
        clf.fit([x[0] for x in train_data], [x[1] for x in train_data])
        result['accuracy'] = clf.score([x[0] for x in test_data], [x[1] for x in test_data])
        print('Accuracy: ' + str(result['accuracy']))
    return result
