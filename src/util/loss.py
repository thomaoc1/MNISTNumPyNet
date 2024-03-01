import numpy as np


class Loss:
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]