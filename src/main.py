import argparse

import numpy as np
from matplotlib import pyplot as plt

from src.util.activation import Activation
from src.dataset import Dataset
from src.nn.nn import NeuralNetwork


def display_example(nn, X_test):
    indices = np.random.choice(range(len(X_test)), 4, replace=False)
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    for i, ax in enumerate(axs.flatten()):
        index = indices[i]
        image = X_test[index].reshape(28, 28)
        predicted_label = nn.predict(X_test[index])

        ax.imshow(image, cmap='gray')
        ax.set_title(f"Prediction: {predicted_label[0]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    dataset = Dataset()
    X_train, y_train = dataset.X_train, dataset.y_train
    X_test, y_test = dataset.X_test, dataset.y_test

    nn = NeuralNetwork(784, *((output, Activation.relu) for output in args.hidden_layers), (10, Activation.softmax))
    nn.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
    nn.evaluate(X_test, y_test)

    display_example(nn, X_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hl', nargs='+', type=int,
                        default=[200, 100], help='Hidden layers ex: 200 100', dest='hidden_layers')

    parser.add_argument('-lr', type=float, default=0.01, help='Learning rate', dest='learning_rate')
    parser.add_argument('-bz', type=int, default=32, help='Batch size', dest='batch_size')
    parser.add_argument('-e', type=int, default=5, help='Epochs', dest='epochs')
    args = parser.parse_args()

    main()
