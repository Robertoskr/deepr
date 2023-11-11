from .base import Layer
import numpy as np
import scipy as sc


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.input_depth = input_depth
        # we will output self.depth matrices. Each a combination of a kernel with X. Each kernel has the same dims
        # as X.
        self.output_height = input_height - self.kernel_size + 1
        self.output_width = input_width - self.kernel_size + 1
        self.output_shape = (self.depth, self.output_height, self.output_width)
        self.kernels_shape = (
            self.depth,
            input_depth,
            self.kernel_size,
            self.kernel_size,
        )

        self.kernels = np.zeros(self.kernels_shape)
        self.biases = np.random.randn(1, self.depth)

    def forward(self, X, is_training: bool = False, *args, **kwargs):
        self.X = X.T
        n = len(self.X)

        # Initialize output volume
        self.output = np.zeros((n, self.depth, self.output_height, self.output_width))

        for image in range(n):
            for j in range(self.input_depth):
                for d in range(self.depth):
                    image_channel = self.X[image, j]
                    kernel = self.kernels[d, j]
                    self.output[image, d, :, :] += sc.signal.correlate2d(
                        image_channel, kernel, mode="valid"
                    )

        return self.output.T

    def backward(self, d_out, learning_rate):
        # d_out is the gradient of the loss with respect to the output of this layer
        # The shape of d_out would be the same as self.output
        n = len(self.X)
        d_out = d_out.T  # Transpose back to the shape used in forward pass

        d_filters = np.zeros_like(self.kernels)

        for image in range(n):
            for j in range(self.input_depth):
                for d in range(self.depth):
                    image_channel = self.X[image, j, :, :]
                    kernel_output_d = d_out[image, d]

                    d_filters[d, j] += sc.signal.correlate2d(
                        image_channel, kernel_output_d, mode="valid"
                    )

        # Update the kernels
        self.kernels = self.kernels - learning_rate * d_filters

        d_X = np.zeros_like(self.X, dtype=np.float32)
        for image in range(n):
            for d in range(self.depth):
                # Rotate the kernel by 180 degrees
                rotated_kernel = np.rot90(self.kernels[d, :, :, :], 2)

                for j in range(self.input_depth):
                    # Convolve with each channel of the output gradient
                    kernel_output_d = d_out[image, d]

                    # Apply convolution using correlate2d
                    d_X[image, j] += (
                        sc.signal.correlate2d(
                            kernel_output_d, rotated_kernel[j], mode="full"
                        )
                        / n
                    )

        return d_X.T
