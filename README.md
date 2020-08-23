# Artificial Neuron

This is an artificial neuron built from scratch. It uses a logistic activation function and a log likelihood objective function. It is used on the Yalefaces dataset and categorizes each face into 1 of 14 different classes.

## Usage

`python3 neuron.py`

## Dependencies

- `python 3.8+`

### Python Dependencies

- `pillow`
- `numpy`
- `matplotlib`
- `seaborn`
- `pandas`

## Hyper Parameters

- Learning Rate: `0.0001`

- Termination Criteria: `100 iterations`

- L2 Regularization Term: `0.5`

- Bias: `1`

- Batch Size: `8`

## Results

### Accuracy:

Testing Accuracy: `95.24%`

### Average Log Likelihood:

![](https://github.com/is386/ArtificialNeuron/blob/master/log.png?raw=true)

### Confusion Matrix:

![](https://github.com/is386/ArtificialNeuron/blob/master/confuse.png?raw=true)
