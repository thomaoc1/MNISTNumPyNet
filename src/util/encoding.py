import numpy as np


class Encoding:
    @staticmethod
    def one_hot_encode(y):
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]