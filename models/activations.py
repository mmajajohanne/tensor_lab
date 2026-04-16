import numpy as np


class Sigmoid:
    """Sigmoid-aktiveringsfunksjon: 1 / (1 + e^(-x))."""

    def __call__(self, x_data):
        return 1 / (1 + np.exp(-x_data))

    def diff(self, x_data):
        s = self(x_data)
        return s * (1 - s)


class ReLU:
    """ReLU-aktiveringsfunksjon: max(0, x)."""

    def __call__(self, x_data):
        return np.maximum(0, x_data)

    def diff(self, x_data):
        return (x_data > 0).astype(float)
