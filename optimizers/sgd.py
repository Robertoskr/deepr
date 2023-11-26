from .base import Optimizer
import numpy as np


class SGD(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = None

    def step(self, params, grads):
        params = list(params)
        grads = list(grads)

        if self.velocities is None:
            self.velocities = []

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.update_fn(grad, param, i)

    def update_fn(self, grad, param, vel_idx):
        if len(self.velocities) < vel_idx + 1:
            self.velocities.append(np.zeros_like(grad))

        self.velocities[vel_idx] = (
            self.momentum * self.velocities[vel_idx] - self.learning_rate * grad
        )

        param += self.velocities[vel_idx]
