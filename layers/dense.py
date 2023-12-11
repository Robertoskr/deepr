from .base import Layer
from ..functions import NoActivation, L2
import numpy as np

"""
x = (n * d)
w = (out * d) 
"""

DEFAULT_REGULARIZATION = L2(0.001)


class Dense(Layer):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        regularization_fn=DEFAULT_REGULARIZATION,
    ):
        self.regularization_fn = regularization_fn
        self.n_input = n_input
        self.n_output = n_output

        self.W = np.random.random((n_output, n_input))
        self.b = np.zeros((n_output, 1))

        # output from the layer (before the activation)
        self.z = None
        self.d_W, self.d_b = None, None
        self.params = ["W", "b"]
        self.grads = ["d_W", "d_b"]

    def _assert_shape_forward(self, X):
        d, n = X.shape
        assert d == self.n_input

    def forward(self, X, is_training: bool = False, *args, **kwargs):
        self.X = X
        self._assert_shape_forward(X)
        self.z = np.dot(self.W, X) + self.b
        return self.z

    def backward(self, prev_grad, is_first_layer):
        d_a = prev_grad
        self.d_W = np.dot(d_a, self.X.T) + self.regularization_fn.derivative(self.W)
        self.d_b = np.sum(d_a, axis=1).reshape(self.n_output, 1)
        return np.dot(self.W.T, d_a)
