from data_extration import load_data
from train import *
from matplotlib import pyplot as plt

train_data, test_data = load_data("samples.data", 0.8)
result = train(train_data, test_data, solver='adam', hidden_layer_sizes=[40, 40], activation='relu', loss_curve_iterations=10, data_percentage_accuracy=True)
#draw plots using result['loss_curve'] and result['data_percentage_accuracy']
plt.plot(result['loss_curve'])
plt.ylabel('loss')
plt.xlabel('iteration')
plt.title('Hidden layers: (414, 414), ReLU, samples data')
#round accuracy to 2 decimal places
plt.savefig('414_414_ReLU_samples_loss_curve.png')
plt.clf()

#draw plot using result['data_percentage_accuracy']
plt.plot(DATA_PERCENTAGE, result['data_percentage_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('data percentage')
plt.title('Hidden layers: (414, 414), ReLU, samples data')
plt.savefig('414_414_ReLU_samples_accuracy_data_percentage.png')
print('Accuracy: ' + str(result['accuracy']))
print('Time: ' + str(result['time']))
plt.clf()

