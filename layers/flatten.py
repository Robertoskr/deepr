from .base import Layer
import numpy as np


class Flatten(Layer):
    def forward(self, X, *args, **kwargs):
        self.original_shape = X.shape
        return X.reshape(-1, X.shape[-1])

    def backward(self, prev_grad, *args, **kwargs):
        shape = tuple(reversed(self.original_shape))
        res = np.zeros(shape)
        for i, X_i in enumerate(prev_grad.T):
            res[i] = X_i.reshape(shape[1:4])

        return res.T
