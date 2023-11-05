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
        activation_fn=NoActivation(),
        regularization_fn=DEFAULT_REGULARIZATION,
    ):
        self.activation = activation_fn
        self.regularization_fn = regularization_fn 
        self.n_input = n_input
        self.n_output = n_output

        self.W = np.random.random((n_output, n_input))
        self.b = np.zeros((n_output, 1))

        # output from the layer (before the activation)
        self.z = None
        # output from the layer (after the activation)

    def _assert_shape_forward(self, X):
        d, n = X.shape
        assert d == self.n_input

    def forward(self, X, is_training: bool = False, *args, **kwargs):
        self.X = X
        self._assert_shape_forward(X)
        self.z = np.dot(self.W, X) + self.b
        self.a = self.activation.base(self.z)
        return self.a

    def _update_weights(self, d_w, d_b, learning_rate):
        self.W = self.W - learning_rate * d_w
        self.b = self.b - learning_rate * d_b

    def backward(self, prev_grad, learning_rate):
        d_a = prev_grad * self.activation.derivative(self.z)
        d_w = np.dot(d_a, self.X.T) + self.regularization_fn.derivative(self.W)
        d_b = np.sum(d_a, axis=1).reshape(self.n_output, 1)
        self._update_weights(d_w, d_b, learning_rate)
        return np.dot(self.W.T, d_a)

