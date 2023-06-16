import joblib

from constants import INSTRUMENTS
from data_extration import load_data
from logistic_regression_n_dim import LogisticRegressionNDim

LEARNING_RATE = 0.01
N_ITERS = 100000
REG_COEF = 0.01

models = {}
train_data, test_data = load_data("../data/cropped_samples_fft_ no_shift.data", 20000, 0.8)

model = LogisticRegressionNDim(len(INSTRUMENTS), LEARNING_RATE, N_ITERS)
model.train([x[0] for x in train_data], [x[1] for x in train_data], REG_COEF)
joblib.dump(model, "../trained_models/logistic_regression_cropped_fft.pkl")
print(model.score([x[0] for x in test_data], [x[1] for x in test_data]))
