import pandas as pd

from src.util.encoding import Encoding


class Dataset:
    def __init__(self, train_fp='datasets/MNIST/mnist_train.csv', test_fp='datasets/MNIST/mnist_test.csv'):
        train_df = pd.read_csv(train_fp)
        test_df = pd.read_csv(test_fp)

        self.y_train = train_df['label'].values
        self.X_train = train_df.drop('label', axis=1).values

        self.y_test = test_df['label'].values
        self.X_test = test_df.drop('label', axis=1).values

        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

        self.y_train = Encoding.one_hot_encode(self.y_train)
        self.y_test = Encoding.one_hot_encode(self.y_test)
