import random
from pprint import pprint
import os
from scipy.io import wavfile

import joblib
from sklearn.neural_network import MLPClassifier

from constants import CHUNK_SIZE, INSTRUMENTS
from data_extration import load_data
print(INSTRUMENTS)

NEURAL_NETWORK_NAME = 'neural_network_multiple_shifts.pkl'
DATA_NAME = 'shifted_multiple_balanced.data'
# HIDDEN_LAYERS = (828,)
# HIDDEN_LAYERS = (200, 200)
HIDDEN_LAYERS = (200, 200)
FEATURES = CHUNK_SIZE
MAX_INSTRUMENTS = 10

train_data, test_data = load_data("samples.data", 0)
# test_data = [x for x in test_data if x[1][5] == 1]

try:
    clf = joblib.load(NEURAL_NETWORK_NAME)
except FileNotFoundError:
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=HIDDEN_LAYERS, random_state=1, activation='relu', verbose=True, warm_start=True, max_iter=1)

wf,data = wavfile.read("samples/Flute/ordinario/Fl-ord-B3-ff-N-N.wav")
test_data = []
for chunk_id in range(len(data) // CHUNK_SIZE):
    chunk = data[chunk_id * CHUNK_SIZE: (chunk_id + 1) * CHUNK_SIZE]
    chunk = chunk.astype(float)
    chunk /= 32768
    chunk /= 14
    print(chunk)
    print(clf.predict([chunk]))
    test_data.append(chunk)

ans = [0] * len(INSTRUMENTS)
ans[5] = 1
print("total: " + str(clf.score([x[0] for x in test_data], [x[1] for x in test_data]) * 100) + '%')


while False:
    clf.fit([x[0] for x in train_data], [x[1] for x in train_data])
    joblib.dump(clf, NEURAL_NETWORK_NAME)
    #score on data with different amount of instruments
    # for i in range(0, MAX_INSTRUMENTS+1):
    #     if test_data[i]:
    #         print(str(i) + ' instruments (' + str(len(test_data[i])) + '): ' + str(clf.score([x[0] for x in test_data[i]], [x[1] for x in test_data[i]]) * 100) + '%')
    #extract x and y from test_data
    print("total: " + str(clf.score([x[0] for x in test_data], [x[1] for x in test_data]) * 100) + '%')

