from abc import ABC
from typing import Generic, List

import numpy as np

from networks.network import Network, T

import torch
from torch import Tensor


class TorchNetwork(Network[Tensor], ABC):

    def __init__(self, architecture, lr, reg):
        super().__init__(architecture, lr, reg)
        self.optim = torch.optim.Adam(self.layers + self.biases, lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def init_layer(self, prev_width, width) -> T:
        return torch.randn(prev_width, width, dtype=torch.float64, requires_grad=True)

    def init_bias(self, width):
        return torch.randn(1, width, dtype=torch.float64, requires_grad=True)

    def softmax(self, inputs):
        return inputs.softmax(dim=1)

    def clamp(self, inputs, minimum=None, maximum=None):
        return inputs.clamp(min=minimum)

    def xw_plus_b(self, X, W, b):
        return X.mm(W) + b

    def perceptronise(self, inputs: Tensor, thresholds: Tensor) -> Tensor:
        return (inputs > thresholds).type(torch.float32)

    def reshape(self, inputs: T, new_shape: List[int]) -> T:
        return inputs.view(*new_shape)

    def backward(self, forward_results: List[Tensor], y_pred: T) -> None:
        data_loss = self.criterion(forward_results[-2], y_pred)

        reg_loss = 0
        for i in range(self.n_layers):
            reg_loss += torch.sum(self.layers[i] * self.layers[i]) * (1 / self.n_layers) * self.reg
        loss = data_loss + reg_loss

        loss.backward()
        with torch.no_grad():
            self.optim.step()
            self.optim.zero_grad()

    @staticmethod
    def to_np(inputs: Tensor) -> np.array:
        return inputs.detach().numpy()

    @staticmethod
    def from_np(inputs: np.array) -> Tensor:
        return torch.from_numpy(inputs)

    def save(self, path: str) -> None:
        pass

    @staticmethod
    def load(path: str) -> Generic[T]:
        pass


