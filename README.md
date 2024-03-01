# Neural-Network from scratch using numpy

This is a simple implementation of a neural network from scratch using numpy. The neural network is trained on the MNIST dataset. The neural network uses backpropagation to update the weights.

## Results
Training the model with 3 hidden layers of 200, 100, and 50 neurons respectively, a learning rate of 0.01, a batch size of 32, and 5 epochs, the model achieved an accuracy of ~96% on the test set.

This is what the training process looks like:

![training.gif](res%2Ftraining.gif)

The following is an example of the model's predictions on the test set:

![example.png](res%2Fexample.png)
## Getting Started
This is if you would like to play with it yourself.

### Prerequisites
Download the MNIST dataset from [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). Extract the files and place them in a directory at the root level named `dataset`.

### Installing
To install the required packages, run the following command:
```
python -m venv venv
pip3 install -r requirements.txt
```

### Running
```
python3 -m src.main [-h] [-hl HIDDEN_LAYERS [HIDDEN_LAYERS ...]] [-lr LEARNING_RATE] [-bz BATCH_SIZE] [-e EPOCHS]

options:
  -h, --help            show this help message and exit
  -hl HIDDEN_LAYERS [HIDDEN_LAYERS ...]
                        Hidden layers ex: 200 100
  -lr LEARNING_RATE     Learning rate
  -bz BATCH_SIZE        Batch size
  -e EPOCHS             Epochs
```

For example (the default):
```
python3 -m src.main -hl 200 100 -lr 0.01 -bz 32 -e 5
```