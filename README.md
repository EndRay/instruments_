# What it does?:

Best accuracy for one note/instrument: 93%

Best accuracy for 1-4 notes/instruments: 88%

## TODO:
- Stop on validation set
- Boost different neural networks
- Neighbor chunk analysis
- Plots
- Simple UI
- Shifted and unshifted (compare trained models and test shifted data on model trained over unshifted data)
- Logistic Regression


### Data formats to compare:
- Sample
- FFT
- Harmonics (works only for one note/instrument)

### Compare activation functions
- Sigmoid
- ReLU

### Compare solvers
- Adam
- SGD
- LBFGS
- Own

### Compare hidden layers

### How to compare:
- Loss curve
- Time to train
- Hidden layers size
- Data percentage to accuracy
- Accuracy on different instruments