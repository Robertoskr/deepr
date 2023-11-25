from .base import Layer
import numpy as np
import scipy as sc


class Convolutional(Layer):
    def __init__(
        self,
        input_shape: tuple,
        kernel_size: int,
        depth: int,
        stride: int = 1,
        padding: int = 0,
        random_kernels: bool = False,
    ):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.input_depth = input_depth
        self.stride = stride
        self.padding = padding
        # as X.
        # Calculate output dimensions
        self.output_height = ((input_height - kernel_size + 2 * padding) // stride) + 1
        self.output_width = ((input_width - kernel_size + 2 * padding) // stride) + 1
        self.output_shape = (self.depth, self.output_height, self.output_width)

        self.kernels_shape = (self.depth, input_depth, kernel_size, kernel_size)
        if random_kernels:
            self.kernels = np.random.random(self.kernels_shape)
        else:
            self.kernels = np.zeros(self.kernels_shape)

        self.biases = np.zeros((1, self.depth)) + 1

        self.d_kernels = None
        self.d_biases = None

        self.params = ["kernels", "biases"]
        self.grads = ["d_kernels", "d_biases"]

    def _forward_stride(self):
        """
        Forward implementation supporting stride.
        Is slower than the forward method, that does not support stride, and uses scipy.signal.correlate2d function
        """
        n = len(self.X)
        self.output = np.zeros((n, self.depth, self.output_height, self.output_width))

        for image in range(n):
            for d in range(self.depth):
                for h in range(self.output_height):
                    for w in range(self.output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        region = self.X[image, :, h_start:h_end, w_start:w_end]
                        self.output[image, d, h, w] = (
                            np.sum(region * self.kernels[d, :, :, :])
                            + self.biases[0, d]
                        )

        return self.output.T

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
        if self.stride > 1:
            return self._forward_stride()

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

    def _backward_stride(self, d_out, is_first_layer):
        n = len(self.X)

        # Initialize gradients
        d_filters = np.zeros_like(self.kernels)
        d_biases = np.zeros_like(self.biases)
        d_X = np.zeros_like(self.X, dtype=np.float32)

        # Calculate d_filters and d_biases
        for image in range(n):
            for d in range(self.depth):
                for h in range(self.output_height):
                    for w in range(self.output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        region = self.X[image, :, h_start:h_end, w_start:w_end]
                        d_filters[d, :, :, :] += region * d_out[image, d, h, w]
                        d_biases[0, d] += d_out[image, d, h, w]

        # Update the kernels and the biases
        self.d_kernels = d_filters
        self.d_biases = d_biases

        if is_first_layer:
            return

        # Calculate d_X
        for image in range(n):
            for d in range(self.depth):
                # Rotate the kernel by 180 degrees
                rotated_kernel = np.rot90(self.kernels[d, :, :, :], 2)

                for h in range(self.output_height):
                    for w in range(self.output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        # Convolve with each channel of the output gradient
                        kernel_output_d = d_out[image, d, h, w]

                        # Apply convolution using correlate2d
                        d_X[image, :, h_start:h_end, w_start:w_end] += (
                            sc.signal.correlate2d(
                                kernel_output_d, rotated_kernel, mode="full"
                            )
                            / n
                        )
        d_X = self._remove_padding(d_X)
        return d_X.T

    def backward(self, d_out, is_first_layer):
        # d_out is the gradient of the loss with respect to the output of this layer
        # The shape of d_out would be the same as self.output
        n = len(self.X)
        d_out = d_out.T  # Transpose back to the shape used in forward pass

        if self.stride > 1:
            return self._backward_stride(d_out, is_first_layer)

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
        self.d_kernels = d_filters
        self.d_biases = d_biases

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
                    d_X[image, j] += (
                        sc.signal.correlate2d(
                            kernel_output_d, rotated_kernel[j], mode="full"
                        )
                        / n
                    )

        d_X = self._remove_padding(d_X)
        return d_X.T
