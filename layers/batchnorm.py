from .base import Layer
import numpy as np
from ..utils import get_data_dimensions, get_dimension_index


class BatchNormalization(Layer):
    def __init__(self):
        # gamma and beta parameters for every dimension, one per channel in images.
        # initialized based on the data in the forward pass.
        self.g = None
        self.b = None
        # mean for every dimension.
        self.mean = None
        # variance for every dimension.
        self.var = None
        self.d_g = None
        self.d_b = None

        self.params = ["g", "b"]
        self.grads = ["d_g", "d_b"]

    def reshape_for_operations(self, var):
        index = get_dimension_index(self.X)
        shape = tuple(
            ((1 if i != index else self.X.shape[i]) for i in range(len(self.X.shape)))
        )
        return var.reshape(shape)

    def forward(self, X, *args, **kwargs):
        if self.g is None:
            dims = get_data_dimensions(X)
            self.g = np.ones(dims)
            self.b = np.zeros(dims)

        self.X = X

        axis = tuple((i for i in range(len(X.shape)) if i != get_dimension_index(X)))

        self.mean = self.reshape_for_operations(np.mean(self.X, axis=axis))
        self.var = self.reshape_for_operations(np.var(self.X, axis=axis))

        X_norm = (self.X - self.mean) / np.sqrt(self.var + 1e-8)
        return X_norm * self.reshape_for_operations(
            self.g
        ) + self.reshape_for_operations(self.b)

    def backward(self, prev_grad, is_first_layer: bool, **kwargs):
        axis = tuple(
            (
                i
                for i in range(len(prev_grad.shape))
                if i != get_dimension_index(prev_grad)
            )
        )
        self.d_b = np.sum(prev_grad, axis=axis)
        self.d_g = np.sum(prev_grad * self.X, axis=axis)

        if is_first_layer:
            return

        n = prev_grad.shape[-1]
        g = self.reshape_for_operations(self.g)

        dX = (
            (1.0 / n)
            * g
            * (self.var + 1e-8) ** (-1.0 / 2.0)
            * (
                n * prev_grad
                - np.sum(prev_grad, axis=0)
                - (self.X - self.mean)
                * (self.var + 1e-8) ** (-1.0)
                * np.sum(prev_grad * (self.X - self.mean), axis=0)
            )
        )

        return dX
