import random
from pprint import pprint

import joblib
from sklearn.neural_network import MLPClassifier

from constants import CHUNK_SIZE, INSTRUMENTS
from data_extration import load_data

NEURAL_NETWORK_NAME = 'neural_network_multiple_shifts.pkl'
DATA_NAME = 'shifted_multiple_balanced.data'
# HIDDEN_LAYERS = (828,)
# HIDDEN_LAYERS = (200, 200)
HIDDEN_LAYERS = (414, 414)
FEATURES = CHUNK_SIZE
MAX_INSTRUMENTS = 10

train_data, test_data = load_data("samples.data", 0.8)

try:
    clf = joblib.load(NEURAL_NETWORK_NAME)
except FileNotFoundError:
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=HIDDEN_LAYERS, random_state=1, activation='relu', verbose=True, warm_start=True, max_iter=1)


while True:
    clf.fit([x[0] for x in train_data], [x[1] for x in train_data])
    # joblib.dump(clf, NEURAL_NETWORK_NAME)
    # score on data with different amount of instruments
    # for i in range(0, MAX_INSTRUMENTS+1):
    #     if test_data[i]:
    #         print(str(i) + ' instruments (' + str(len(test_data[i])) + '): ' + str(clf.score([x[0] for x in test_data[i]], [x[1] for x in test_data[i]]) * 100) + '%')
    #extract x and y from test_data
    print("total: " + str(clf.score([x[0] for x in train_data], [x[1] for x in train_data]) * 100) + '%')

