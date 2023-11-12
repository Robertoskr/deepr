from .base import Layer
import numpy as np


class MaxPooling(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X, *args, **kwargs):
        self.X = X.T
        n, depth, height, width = self.X.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        pooled = np.zeros((n, depth, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                pooled[:, :, i, j] += np.max(
                    self.X[:, :, h_start:h_end, w_start:w_end], axis=(2, 3)
                )

        return pooled.T

    def backward(self, prev_grad, *args, **kwargs):
        prev_grad = prev_grad.T
        n, depth, height, width = self.X.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        d_X = np.zeros_like(self.X)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                # find out the index of the max value in each region
                max_indices = np.argmax(
                    self.X[:, :, h_start:h_end, w_start:w_end].reshape(n, depth, -1),
                    axis=2,
                )
                # Convert the 1D indices to 2D coordinates
                h_max, w_max = np.unravel_index(
                    max_indices, (self.pool_size, self.pool_size)
                )

                # Create a mask for each image and each depth channel
                for image in range(n):
                    for channel in range(depth):
                        d_X[image, channel, h_start:h_end, w_start:w_end][
                            h_max[image, channel], w_max[image, channel]
                        ] += prev_grad[image, channel, i, j]

        return d_X.T
