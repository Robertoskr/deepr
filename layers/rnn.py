from deepr.functions import Sigmoid
from .base import Layer
import numpy as np

"""
Basic Recurrent neural network layer block implementation. 
"""


class RNN(Layer):
    def __init__(
        self,
        seq_length,
        n_inputs,
        n_hidden,
        n_outputs,
        activation,
        return_sequences: bool = False,
    ):
        self.seq_length = seq_length
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.activation = activation
        self.return_sequences = return_sequences

        # Initialize weights and biases
        self.Wx = np.random.randn(n_inputs, n_hidden) * 0.01
        self.Wh = np.random.randn(n_hidden, n_hidden) * 0.01
        self.Wo = np.random.randn(n_hidden, n_outputs) * 0.01
        self.bx = np.zeros((1, n_hidden))
        self.bo = np.zeros((1, n_outputs))

        self.params = ["Wx", "Wh", "Wo", "bx", "bo"]
        self.grads = ["dWx", "dWh", "dWo", "dbx", "dbo"]

    def forward(self, X, *args, **kwargs):
        self.X = X.T
        self.H = np.zeros((self.X.shape[0], self.seq_length, self.n_hidden))
        self.O = np.zeros((self.X.shape[0], self.seq_length, self.n_outputs))

        for t in range(self.seq_length):
            X_t = self.X[:, t, :]
            if t > 0:
                H_t = self.activation.base(
                    X_t @ self.Wx + self.H[:, t - 1, :] @ self.Wh + self.bx
                )
            else:
                H_t = self.activation.base(X_t @ self.Wx + self.bx)

            self.H[:, t, :] = H_t
            self.O[:, t, :] = H_t @ self.Wo + self.bo

        return (self.O[:, -1, :] if not self.return_sequences else self.O).T

    def backward(self, dO, is_first_layer=False):
        dO = dO.T
        if not self.return_sequences:
            # Extend dO to have the same shape as self.O, but with zeros for all time steps except the last one
            dO_extended = np.zeros_like(self.O)
            dO_extended[:, -1, :] = dO
            dO = dO_extended

        dWx, dWh, dWo = (
            np.zeros_like(self.Wx),
            np.zeros_like(self.Wh),
            np.zeros_like(self.Wo),
        )
        dbx, dbo = np.zeros_like(self.bx), np.zeros_like(self.bo)
        dH_next = np.zeros((dO.shape[0], self.n_hidden))

        for t in reversed(range(self.seq_length)):
            dO_t = dO[:, t, :]
            dH_t = dO_t @ self.Wo.T + dH_next
            dH_raw = dH_t * self.activation.derivative(self.H[:, t, :])

            if t > 0:
                dWh += self.H[:, t - 1, :].T @ dH_raw
            dbx += np.sum(dH_raw, axis=0, keepdims=True)
            dWx += self.X[:, t, :].T @ dH_raw
            dH_next = dH_raw @ self.Wh.T
            dWo += self.H[:, t, :].T @ dO_t
            dbo += np.sum(dO_t, axis=0, keepdims=True)

        self.dWx, self.dWh, self.dWo = dWx, dWh, dWo
        self.dbx, self.dbo = dbx, dbo

        for param, grad in zip(self.params, self.grads):
            self.__dict__[grad] = -self.__dict__[grad]
