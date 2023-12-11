from .base import Layer
from ..functions import NoActivation, Sigmoid
import numpy as np


class LSTM(Layer):
    def __init__(
        self,
        n_inputs,
        seq_length,
        n_hidden,
        n_outputs,
        activation=None,
        return_sequences: bool = False,
    ):
        self.n_inputs = n_inputs
        self.seq_length = seq_length
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.sigmoid = Sigmoid()
        self.return_sequences = return_sequences
        self.activation = activation or NoActivation()

        # Forget gate weight (n_inputs + n_hidden) -> (n_hidden)
        self.Wf = np.random.randn(n_inputs + n_hidden, n_hidden) * 0.01
        # Input gate weight (n_inputs + n_hidden) -> (n_hidden)
        self.Wi = np.random.randn(n_inputs + n_hidden, n_hidden) * 0.01
        # Candidate cell state weight (n_inputs + n_hidden) -> (n_hidden)
        self.Wc = np.random.randn(n_inputs + n_hidden, n_hidden) * 0.01
        # Output gate weight (n_inputs + hidden) -> (n_hidden)
        self.Wo = np.random.randn(n_inputs + n_hidden, n_hidden) * 0.01
        # Output weight (n_hidden) -> (n_outputs)
        self.Wy = np.random.randn(n_hidden, n_outputs) * 0.01
        # biases
        self.bf = np.zeros((1, n_hidden))
        self.bi = np.zeros((1, n_hidden))
        self.bc = np.zeros((1, n_hidden))
        self.bo = np.zeros((1, n_hidden))
        self.by = np.zeros((1, n_outputs))

        self.params = ["Wf", "Wi", "Wc", "Wo", "Wy", "bf", "bi", "bc", "bo", "by"]
        self.grads = [
            "dWf",
            "dWi",
            "dWc",
            "dWo",
            "dWy",
            "dbf",
            "dbi",
            "dbc",
            "dbo",
            "dby",
        ]

    def forward(self, X, *args, **kwargs):
        self.X = X.T
        self.y = np.zeros((self.X.shape[0], self.seq_length, self.n_outputs))
        for t in range(self.seq_length):
            X_t = self.X[:, t, :]
            if t == 0:
                self.c = np.zeros((X_t.shape[0], self.n_hidden))
                self.a = np.zeros((X_t.shape[0], self.n_hidden))

            X_t = np.hstack((X_t, self.a))

            # Forget gate
            self.f = self.sigmoid.forward(X_t @ self.Wf + self.bf)

            # Input gate
            self.i = self.sigmoid.forward(X_t @ self.Wi + self.bi)

            # Candidate cell state
            self.c_ = np.tanh(X_t @ self.Wc + self.bc)

            # Update cell state
            self.c = self.f * self.c + self.i * self.c_

            # Output gate
            self.o = self.sigmoid.forward(X_t @ self.Wo + self.bo)

            # Update hidden state
            self.a = self.o * np.tanh(self.c)
            self.y_ = self.a @ self.Wy + self.by
            self.y[:, t, :] = self.activation.base(self.y_)

        if self.return_sequences:
            return self.y.T
        return self.y[:, -1, :].T

    def backward(self, dY, *args, **kwargs):
        dY = dY.T
        # Initialize gradients for each parameter
        dWf, dWi, dWc, dWo, dWy = [
            np.zeros_like(w) for w in (self.Wf, self.Wi, self.Wc, self.Wo, self.Wy)
        ]
        dbf, dbi, dbc, dbo, dby = [
            np.zeros_like(b) for b in (self.bf, self.bi, self.bc, self.bo, self.by)
        ]

        # Initialize gradients w.r.t. hidden state and cell state
        da_next, dc_next = np.zeros_like(self.a[0]), np.zeros_like(self.c)

        # Backpropagation through time
        for t in reversed(range(self.seq_length)):
            if not self.return_sequences:
                dY_t = dY if t == self.seq_length - 1 else np.zeros_like(dY)
            else:
                dY_t = dY[:, t, :]

            # Gradient w.r.t. output
            da = dY_t * self.activation.derivative(self.y_) @ self.Wy.T + da_next
            dby += np.sum(dY_t, axis=0, keepdims=True)
            dWy += self.a.T @ dY_t

            # Gradient w.r.t. hidden state
            do = da * np.tanh(self.c)
            do = do * self.sigmoid.derivative(self.o)
            dbo += np.sum(do, axis=0, keepdims=True)
            dWo += np.hstack((self.X[:, t, :], self.a)).T @ do

            # Gradient w.r.t. cell state
            dc = da * self.o * (1 - np.tanh(self.c) ** 2) + dc_next
            dc_ = dc * self.i
            dc_ = dc_ * (1 - self.c_**2)
            dbc += np.sum(dc_, axis=0, keepdims=True)
            dWc += np.hstack((self.X[:, t, :], self.a)).T @ dc_

            # Gradient w.r.t. input gate
            di = dc * self.c_
            di = di * self.sigmoid.derivative(self.i)
            dbi += np.sum(di, axis=0, keepdims=True)
            dWi += np.hstack((self.X[:, t, :], self.a)).T @ di

            # Gradient w.r.t. forget gate
            df = dc * self.c
            df = df * self.sigmoid.derivative(self.f)
            dbf += np.sum(df, axis=0, keepdims=True)
            dWf += np.hstack((self.X[:, t, :], self.a)).T @ df

            # Gradient w.r.t. input at time t
            dX_t = df @ self.Wf.T + di @ self.Wi.T + dc_ @ self.Wc.T + do @ self.Wo.T

            # Update gradients w.r.t. next hidden state and next cell state
            da_next = dX_t[:, self.n_inputs :]
            dc_next = self.f * dc

        # Store gradients
        self.dWf, self.dWi, self.dWc, self.dWo, self.dWy = dWf, dWi, dWc, dWo, dWy
        self.dbf, self.dbi, self.dbc, self.dbo, self.dby = dbf, dbi, dbc, dbo, dby

        for grad in self.grads:
            self.__dict__[grad] = -self.__dict__[grad]

        # Return gradient w.r.t. input for further backpropagation
        return dX_t[:, : self.n_inputs]
