from .base import Layer
import numpy as np
import scipy as sc


class Convolutional(Layer):
    def __init__(
        self,
        input_shape: tuple,
        kernel_size: int,
        depth: int,
        padding: int = 0,
        random_kernels: bool = False,
    ):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.input_depth = input_depth
        self.padding = padding
        # as X.
        # Calculate output dimensions
        self.output_height = input_height - kernel_size + 1
        self.output_width = input_width - kernel_size + 1
        self.output_height = (input_height - kernel_size + 2 * padding) + 1
        self.output_width = (input_width - kernel_size + 2 * padding) + 1
        self.output_shape = (self.depth, self.output_height, self.output_width)

        self.kernels_shape = (self.depth, input_depth, kernel_size, kernel_size)
        if random_kernels:
            self.kernels = np.random.normal(0, 0.01, self.kernels_shape)
        else:
            self.kernels = np.zeros(self.kernels_shape)

        self.biases = np.zeros((1, self.depth)) + 1

        self.d_kernels = None
        self.d_biases = None

        self.params = ["kernels", "biases"]
        self.grads = ["d_kernels", "d_biases"]

    def forward(self, X, is_training: bool = False, *args, **kwargs):
        self.X = np.pad(
            X.T,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="constant",
        )
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

                # Add the bias for this filter
                self.output[image, d, :, :] += self.biases[0, d]

        return self.output.T

    def _remove_padding(self, X):
        # Remove padding from d_X if padding was used
        if self.padding > 0:
            X = X[:, :, self.padding : -self.padding, self.padding : -self.padding]
        return X

    def backward(self, d_out, is_first_layer):
        # d_out is the gradient of the loss with respect to the output of this layer
        # The shape of d_out would be the same as self.output
        n = len(self.X)
        d_out = d_out.T  # Transpose back to the shape used in forward pass

        d_filters = np.zeros_like(self.kernels)
        d_biases = np.zeros_like(self.biases)

        for image in range(n):
            for j in range(self.input_depth):
                for d in range(self.depth):
                    image_channel = self.X[image, j, :, :]
                    kernel_output_d = d_out[image, d]

                    d_filters[d, j] += sc.signal.correlate2d(
                        image_channel, kernel_output_d, mode="valid"
                    )

        # Initialize gradient for biases
        d_biases = np.zeros_like(self.biases)

        for d in range(self.depth):
            for image in range(n):
                # Sum gradients for each filter across all positions
                d_biases[0, d] += np.sum(d_out[image, d])

        # Update the kernels and the biases
        self.d_kernels = -d_filters
        self.d_biases = -d_biases

        if is_first_layer:
            # Early exit, there is nothing more to optimize
            return

        d_X = np.zeros_like(self.X, dtype=np.float32)
        for image in range(n):
            for d in range(self.depth):
                # Rotate the kernel by 180 degrees
                rotated_kernel = np.rot90(self.kernels[d, :, :, :], 2)

                for j in range(self.input_depth):
                    # Convolve with each channel of the output gradient
                    kernel_output_d = d_out[image, d]

                    # Apply convolution using correlate2d
                    d_X[image, j] += sc.signal.correlate2d(
                        kernel_output_d, rotated_kernel[j], mode="full"
                    )

        d_X = self._remove_padding(d_X)
        return d_X.T
