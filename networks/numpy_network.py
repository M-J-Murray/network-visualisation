from typing import Generic

from networks.network import Network, List, T
import numpy as np


class NumpyNetwork(Network[np.ndarray]):

    def __init__(self, architecture, lr, reg):
        super().__init__(architecture, lr, reg)

    def init_layer(self, prev_width, width):
        return np.random.randn(prev_width, width).astype("float32")

    def init_bias(self, width):
        return np.zeros((1, width)).astype("float32")

    def softmax(self, inputs):
        exp_f = np.exp(inputs)
        return exp_f / np.sum(exp_f, axis=1, keepdims=True)

    def clamp(self, inputs, minimum=None, maximum=None):
        result = inputs
        if minimum is not None:
            result = np.maximum(result, minimum)
        if maximum is not None:
            result = np.minimum(result, maximum)
        return result

    def xw_plus_b(self, X, W, b):
        return np.matmul(X, W) + b

    def perceptronise(self, inputs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        return (inputs > thresholds).astype("float32")

    def reshape(self, inputs: T, new_shape: List[int]) -> T:
        return np.reshape(inputs, newshape=new_shape)

    def backward(self, forward_results: List[np.ndarray], y_pred: np.ndarray) -> None:
        error = forward_results[-1]
        error[range(error.shape[0]), y_pred] -= 1
        error /= error.shape[0]

        for i in reversed(range(self.n_layers)):
            self.layers[i] -= self.lr * self.reg * np.dot(forward_results[i*2].T, error)
            self.biases[i] -= self.lr * np.sum(error, axis=0, keepdims=True)

            error = np.dot(error, self.layers[i].T)
            error[forward_results[i*2] <= 0] = 0

        # dscores = forward_results[-1]
        # dscores[range(error.shape[0]), y_pred] -= 1
        # dscores /= error.shape[0]
        #
        # dhidden = np.dot(dscores, self.layers[1].T)
        # dhidden[forward_results[2] <= 0] = 0
        #
        # self.layers[0] -= self.lr * self.reg * np.dot(forward_results[0].T, dhidden)
        # self.biases[0] -= self.lr * np.sum(dhidden, axis=0, keepdims=True)
        # self.layers[1] -= self.lr * self.reg * np.dot(forward_results[2].T, dscores)
        # self.biases[1] -= self.lr * np.sum(dscores, axis=0, keepdims=True)

    @staticmethod
    def to_np(inputs: np.ndarray) -> np.ndarray:
        return inputs

    @staticmethod
    def from_np(inputs: np.ndarray) -> np.ndarray:
        return inputs

    def save(self, path: str) -> None:
        pass

    @staticmethod
    def load(path: str) -> Generic[T]:
        pass


