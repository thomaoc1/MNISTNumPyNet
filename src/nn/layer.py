import numpy as np
from src.util.activation import Activation


class Layer:
    def __init__(self, input_size, output_size, activation=Activation.relu, output_layer=False):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.b = np.zeros((1, output_size))
        self.activation = activation
        self.output_layer = output_layer
        self.z = None
        self.a = None

    def forward(self, X):
        self.z = np.dot(X, self.W.T) + self.b
        self.a = self.activation(self.z)
        return self.a