from .base import Optimizer
import numpy as np


class RMSProp(Optimizer):
    def __init__(self, learning_rate: float, decay_rate: float = 0.9):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = 1e-8
        self.betas = []

    def update_fn(self, grad, param):
        if len(self.betas) < self.index + 1:
            self.betas.append(np.zeros_like(grad))

        self.betas[self.index] = self.decay_rate * self.betas[self.index] + (
            1 - self.decay_rate
        ) * (grad * grad)

        param += (
            -self.learning_rate
            * grad
            / (np.sqrt(self.betas[self.index] + self.epsilon))
        )
