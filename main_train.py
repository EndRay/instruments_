import random
from pprint import pprint
import os
from scipy.io import wavfile

import joblib
from sklearn.neural_network import MLPClassifier

from constants import CHUNK_SIZE, INSTRUMENTS
from data_extration import load_data, SAMPLES, extract_data

print(INSTRUMENTS)

NEURAL_NETWORK_NAME = 'ddd.pkl'
DATA_NAME = 'shifted_multiple_balanced.data'
# HIDDEN_LAYERS = (828,)
HIDDEN_LAYERS = (200, 200)
FEATURES = CHUNK_SIZE
MAX_INSTRUMENTS = 10

# extract_data('trash', 10000)

train_data, test_data = load_data("cropped_samples_fft.data", 0.8)
# test_data = [x for x in test_data if x[1][5] == 1]

try:
    clf = joblib.load(NEURAL_NETWORK_NAME)
except FileNotFoundError:
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=HIDDEN_LAYERS, random_state=1, activation='relu', verbose=True, warm_start=True, max_iter=1)

# for i in range(len(SAMPLES)):
#     sample = SAMPLES[i]
#
#     chunks = [sample[i][0] for i in range(len(sample))]
#     instruments = [sample[i][1] for i in range(len(sample))]
#
#     print(str(clf.score(chunks, instruments) * 100) + '%')
#
# for i in range(len(INSTRUMENTS)):
#     print(INSTRUMENTS[i] + ': ' + str(sum([x[1][i] for x in test_data])))

# print("total: " + str(clf.score([x[0] for x in test_data], [x[1] for x in test_data]) * 100) + '%')
# for i in range(len(INSTRUMENTS)):
#     filtered_test_data = [x for x in test_data if x[1][i] == 1]
#     print(INSTRUMENTS[i] + ': ' + str(clf.score([x[0] for x in filtered_test_data], [x[1] for x in filtered_test_data]) * 100) + '%')

while True:
    clf.fit([x[0] for x in train_data], [x[1] for x in train_data])
    joblib.dump(clf, NEURAL_NETWORK_NAME)
    #score on data with different amount of instruments
    # for i in range(0, MAX_INSTRUMENTS+1):
    #     if test_data[i]:
    #         print(str(i) + ' instruments (' + str(len(test_data[i])) + '): ' + str(clf.score([x[0] for x in test_data[i]], [x[1] for x in test_data[i]]) * 100) + '%')
    #extract x and y from test_data
    print("total: " + str(clf.score([x[0] for x in test_data], [x[1] for x in test_data]) * 100) + '%')

