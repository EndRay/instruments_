# read wav file and spectrum analysis
import random

import joblib
from sklearn.neural_network import MLPClassifier

from constants import INSTRUMENTS

data = []

with open('data.data', 'r') as f:
    for line in f:
        line = line.split(' ')
        data.append([[float(x) for x in line[:21]], [int(x) for x in line[21:]]])

print(len(data))

random.shuffle(data)

train_data = data[:int(len(data) * 0.8)]
test_data = data[int(len(data) * 0.8):]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(828,), random_state=1, activation='logistic', verbose=True)


# train neural network
clf.fit([x[0] for x in train_data], [x[1] for x in train_data])



joblib.dump(clf, 'neural_network%%.pkl')

# test neural network
print(str(clf.score([x[0] for x in test_data], [x[1] for x in test_data]) * 100) + '%')

