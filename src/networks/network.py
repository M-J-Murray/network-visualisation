from abc import abstractmethod
from typing import TypeVar, Generic, List, Set
import numpy as np
from RunningStats import RunningStats

T = TypeVar('T')


def shuffle_dataset(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]
    test_samples = int(X.shape[0] * 0.1)
    train_set = X[test_samples:]
    train_labels = y[test_samples:].astype("long")
    test_set = X[:test_samples]
    test_labels = y[:test_samples].astype("long")

    return train_set, train_labels, test_set, test_labels


class Network(Generic[T]):

    def __init__(self, architecture=None, lr=None, reg=None, layers=None, biases=None):
        super().__init__()
        assert (architecture and lr and reg) or (layers and biases)

        if architecture:
            self.architecture = architecture
            self.n_inputs = architecture[0]
            self.n_classes = architecture[-1]
            self.n_layers = len(architecture) - 1

            self.lr = lr
            self.reg = reg

            self.is_trained = False

            self.layers: List[T] = []
            self.biases: List[T] = []
            for i in range(1, len(architecture)):
                self.layers.append(self.init_layer(architecture[i - 1], architecture[i]))
                self.biases.append(self.init_bias(architecture[i]))
        else:
            self.layers = layers
            self.biases = biases

            if lr is not None and reg is not None:
                self.lr = lr
                self.reg = reg
                self.is_trained = False
            else:
                self.is_trained = True

            self.architecture = [layers[0].shape[0]]
            self.architecture += [layer.shape[1] for layer in layers]
            self.n_inputs = self.architecture[0]
            self.n_classes = self.architecture[-1]
            self.n_layers = len(self.architecture) - 1



    @abstractmethod
    def init_layer(self, prev_width, width) -> T:
        pass

    @abstractmethod
    def init_bias(self, width) -> T:
        pass

    def select_neurons(self, arch_selection: List[Set[int]]) -> (List[T], List[T]):
        selected_layers = []
        selected_biases = []
        if arch_selection:
            assert len(arch_selection) == self.n_layers + 1
            assert len(arch_selection[0]) == self.n_inputs
            for i in range(1, self.n_layers + 1):
                neurons = list(arch_selection[i])
                prev_neurons = list(arch_selection[i - 1])
                selected_layers.append(self.layers[i - 1][prev_neurons, :][:, neurons])
                selected_biases.append(self.biases[i - 1][:, neurons])
        else:
            selected_layers = self.layers
            selected_biases = self.biases

        return selected_layers, selected_biases

    @abstractmethod
    def softmax(self, inputs: T) -> T:
        pass

    @abstractmethod
    def normalise(self, inputs: T) -> T:
        pass

    @abstractmethod
    def clamp(self, inputs: T, minimum: int = None, maximum: int = None) -> T:
        pass

    @abstractmethod
    def xw_plus_b(self, x: T, w: T, b: T) -> T:
        pass

    @abstractmethod
    def perceptronise(self, inputs: T, thresholds: T) -> T:
        pass

    @abstractmethod
    def reshape(self, inputs: T, new_shape: List[int]) -> T:
        pass

    def forward(self, inputs: T, arch_selection: List[Set[int]] = None, precision: int = None, is_mlp: bool = False) -> List[T]:
        sel_layers, sel_biases = self.select_neurons(arch_selection)

        outputs = [inputs]
        for i in range(self.n_layers):
            outputs.append(self.xw_plus_b(outputs[-1], sel_layers[i], sel_biases[i]))
            if i != self.n_layers - 1:
                if is_mlp:
                    outputs.append(self.perceptronise(outputs[-1], -sel_biases[i]))
                else:
                    outputs.append(self.clamp(outputs[-1], minimum=0))

        if outputs[-1].shape[1] == 1:
            outputs.append(self.normalise(outputs[-1]))
        else:
            outputs.append(self.softmax(outputs[-1]))

        if not precision:
            return outputs
        else:
            reshaped = []
            for output in outputs:
                reshaped.append(self.reshape(output, [precision, precision, -1]))
            return reshaped

    @abstractmethod
    def backward(self, forward_results: List[T], y_pred: T) -> None:
        pass

    @staticmethod
    @abstractmethod
    def to_np(inputs: T) -> np.array:
        pass

    @staticmethod
    @abstractmethod
    def from_np(inputs: np.array) -> T:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def load(path: str) -> Generic[T]:
        pass

    @staticmethod
    @abstractmethod
    def load(path: str) -> Generic[T]:
        pass

    @staticmethod
    @abstractmethod
    def save_exists(path: str) -> bool:
        pass

    def train(self, X: np.array, y: np.array, window_size: int, log: bool = True):
        assert not self.is_trained

        stats = RunningStats(window_size)

        n_test_samples = int(X.shape[0] * 0.1)

        episode = 0
        while not stats.finished_window() or stats.window_improved():
            train_set, train_labels, test_set, test_labels = [self.from_np(entry) for entry in shuffle_dataset(X, y)]

            train_pred = self.forward(train_set)
            self.backward(train_pred, train_labels)

            test_pred = self.forward(test_set)
            test_accuracy = (self.to_np(test_pred[-1][range(n_test_samples), test_labels]) > (1 / self.n_classes)).sum() / len(test_labels) * 100
            stats.insert(test_accuracy)

            if log:
                print(f"Episode: {episode}, Test Accuracy: {test_accuracy:6.2f}, Running Avg: {stats.get_average():6.3f}")

        return stats.get_average()
