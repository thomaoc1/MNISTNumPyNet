import numpy as np
from tqdm import tqdm

from src.util.activation import Activation
from src.util.loss import Loss
from src.nn.layer import Layer


class NeuralNetwork:
    def __init__(self, *args):
        """
        :param args: (input_size, (output_size, Activation), (output_size, Activation), ... , (output_size, Activation))
        """
        if not isinstance(args[0], int):
            raise ValueError('First argument must be an integer')

        prev_output_size = args[0]
        layers = args[1:]

        self.layers = []
        for layer in layers:
            if not isinstance(layer[0], int) or not callable(layer[1]):
                raise ValueError('Invalid layer: (output_size (int), Activation (callable)) expected')
            output_size, activation = layer
            self.layers.append(Layer(prev_output_size, output_size, activation))
            prev_output_size = output_size

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        D = np.empty(len(self.layers), dtype=object)
        output_error = self.layers[-1].a - y
        for idx in reversed(range(len(self.layers))):
            layer = self.layers[idx]
            if idx == 0:
                prev_layer_a = X
            else:
                prev_layer_a = self.layers[idx - 1].a

            dW = np.dot(output_error.T, prev_layer_a)
            db = np.sum(output_error, axis=0, keepdims=True)

            output_error = np.dot(output_error, layer.W) * Activation.relu_derivative(prev_layer_a)
            D[idx] = (dW, db)

        for idx, layer in enumerate(self.layers):
            dW, db = D[idx]
            layer.W -= learning_rate * dW / m
            layer.b -= learning_rate * db / m

    def fit(self, X, y, batch_size=32, epochs=10, learning_rate=0.01):
        print('Training with the following configuration:')
        print(f' -> Batch size: {batch_size}')
        print(f' -> Epochs: {epochs}')
        print(f' -> Learning rate: {learning_rate}')
        print()

        num_batches = np.ceil(X.shape[0] / batch_size).astype(int)
        for i in range(epochs):
            print(f'Epoch {i + 1}/{epochs}')
            pbar = tqdm(total=num_batches, leave=True)
            for j in range(0, X.shape[0], batch_size):
                X_batch = X[j:j + batch_size]
                y_batch = y[j:j + batch_size]
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
                pbar.set_description(f'Loss: {Loss.categorical_crossentropy(y_batch, y_pred):.4f}')
                pbar.update(1)
                pbar.refresh()
            pbar.close()
            print()

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == np.argmax(y, axis=1))
        print(f'Accuracy: {accuracy * 100}%')
