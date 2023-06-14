import random
from pprint import pprint

import joblib
from sklearn.neural_network import MLPClassifier

from constants import CHUNK_SIZE, INSTRUMENTS

NEURAL_NETWORK_NAME = 'neural_network_multiple_shifts.pkl'
DATA_NAME = 'shifted_multiple.data'
# HIDDEN_LAYERS = (828,)
# HIDDEN_LAYERS = (200, 200)
HIDDEN_LAYERS = (414, 414)
FEATURES = CHUNK_SIZE
MAX_INSTRUMENTS = 5

data = [[] for _ in range(MAX_INSTRUMENTS+1)]

with open(DATA_NAME, 'r') as f:
    for line in f:
        line = line.split(' ')
        data_line = [[float(x) for x in line[:FEATURES]], [int(x) for x in line[FEATURES:]]]
        cnt = sum(data_line[1])
        data[cnt].append(data_line)

# pprint(data[-1][-1])
train_data = [[] for _ in range(MAX_INSTRUMENTS+1)]
test_data = [[] for _ in range(MAX_INSTRUMENTS+1)]

for i in range(MAX_INSTRUMENTS+1):
    train_data[i] = data[i][:int(len(data[i]) * 0.8)]
    test_data[i] = data[i][int(len(data[i]) * 0.8):]
# pprint(train_data)


# flatten train_data
train_data = [x for y in train_data for x in y]
# shuffle train_data
random.shuffle(train_data)

try:
    clf = joblib.load(NEURAL_NETWORK_NAME)
except FileNotFoundError:
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=HIDDEN_LAYERS, random_state=1, activation='relu', verbose=True, warm_start=True, max_iter=1)


while True:
    clf.fit([x[0] for x in train_data], [x[1] for x in train_data])
    # joblib.dump(clf, NEURAL_NETWORK_NAME)
    # score on data with different amount of instruments
    for i in range(0, MAX_INSTRUMENTS+1):
        if test_data[i]:
            print(str(i) + ' instruments (' + str(len(test_data[i])) + '): ' + str(clf.score([x[0] for x in test_data[i]], [x[1] for x in test_data[i]]) * 100) + '%')
    print("total: " + str(clf.score([x[0] for y in test_data for x in y], [x[1] for y in test_data for x in y]) * 100) + '%')

