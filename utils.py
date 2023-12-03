import numpy as np


def get_dimension_index(X):
    """
    Gets the index where the dimension is located in the data.
    We follow the convention that each sample is a column.
    """
    shape = X.shape
    # tabular data
    if len(shape) == 2:
        return 0
    # images (height, width, channels, n_samples)
    elif len(shape) == 4:
        return 2


def get_data_dimensions(X):
    """
    Gets the number of dimensions of the data.
    For example if the input is d * n, then the number of dimensions is d.
    If the data is images, then the number of dimensions are the channels.
    etc...
    We follow the convention that each sample is a column.
    """
    shape = X.shape
    return shape[get_dimension_index(X)]
