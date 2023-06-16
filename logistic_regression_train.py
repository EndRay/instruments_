from constants import INSTRUMENTS
from data_extration import load_data
from logistic_regression import LogisticRegression
import numpy as np

LEARNING_RATE = 0.05
N_ITERS = 100000
REG_COEF = 0.01

models = {}
train_data, test_data = load_data("cropped_samples_fft.data", 30000, 0.8)
for instrument in INSTRUMENTS:
    models[instrument] = LogisticRegression(LEARNING_RATE, N_ITERS)

for instrument in INSTRUMENTS:
    print("Training model for " + instrument)
    train_data_x = np.array([train_data[i][0] for i in range(len(train_data))])
    train_data_y = np.array([train_data[i][1][INSTRUMENTS.index(instrument)] for i in range(len(train_data))])
    test_data_x = np.array([test_data[i][0] for i in range(len(test_data))])
    test_data_y = np.array([test_data[i][1][INSTRUMENTS.index(instrument)] for i in range(len(test_data))])
    models[instrument].fit(train_data_x, train_data_y, REG_COEF)
    print("Score: ", models[instrument].score(test_data_x, test_data_y))



#score the models
score = 0
for sample in test_data:
    predicted = []
    for instrument in INSTRUMENTS:
        predicted.append(models[instrument].predict(sample[0]))
    if predicted == sample[1]:
        score += 1
print(score / len(test_data) * 100, "%")
