import os
import numpy as np
import torch

from Datasets import Dataset
from RunningStats import RunningStats


def shuffle_dataset(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]
    test_samples = int(X.shape[0]*0.1)
    train_set = X[test_samples:]
    train_labels = y[test_samples:].astype("int32")
    test_set = X[:test_samples]
    test_labels = y[:test_samples].astype("int32")

    return train_set, train_labels, test_set, test_labels


def dataset_to_torch(train_set, train_labels, test_set, test_labels):
    train_labels = torch.from_numpy(train_labels)
    test_labels = torch.from_numpy(test_labels)
    return torch.from_numpy(train_set), train_labels.type(torch.long), torch.from_numpy(test_set), test_labels.type(torch.long)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(f: np.ndarray) -> np.ndarray:
    exp_f = np.exp(f)
    return exp_f / np.sum(exp_f, axis=1, keepdims=True)


def xw_plus_b(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(x, w) + b


def select_neurons(network, p1_enabled) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    if p1_enabled:
        w1 = np.stack([network["W1"][:, i] for i in range(network["W1"].shape[1]) if i in p1_enabled], axis=1)
        b1 = np.stack([network["b1"][:, i] for i in range(network["b1"].shape[1]) if i in p1_enabled], axis=1)
        w2 = np.stack([network["W2"][i, :] for i in range(network["W2"].shape[0]) if i in p1_enabled], axis=0)
        b2 = network["b2"]
    else:
        w1 = network["W1"]
        b1 = network["b1"]
        w2 = network["W2"]
        b2 = network["b2"]

    return w1, b1, w2, b2


def leap(x, threshold: np.ndarray):
    return (x > threshold).astype("float32")


def torch_forward(X, layers, biases, use_softmax):
    result = X
    for i in range(len(layers)):
        result = result.mm(layers[i]) + biases[i]
        if i != len(layers) - 1:
            result = result.clamp(min=0)
    if use_softmax:
        result = result.softmax(dim=1)
    return result


def forward(inputs, network, dataset: Dataset, p1_enabled=None, precision=None, squeeze=False, cut=False):
    w1, b1, w2, b2 = select_neurons(network, p1_enabled)
    out1 = xw_plus_b(inputs, w1, b1)
    if cut:
        out2 = leap(out1, -b1)
    else:
        out2 = np.maximum(out1, 0)
    if squeeze and dataset is Dataset.CIRCLE:
        out3 = np.squeeze(xw_plus_b(out2, w2, b2), axis=1)
    else:
        out3 = xw_plus_b(out2, w2, b2)
    if cut:
        out4 = leap(out3, -b2)
    else:
        if dataset is Dataset.CIRCLE:
            out4 = sigmoid(out3)
        elif dataset is Dataset.SPIRAL:
            out4 = softmax(out3)
        else:
            raise Exception("Invalid dataset type")
    if not precision:
        return out1, out2, out3, out4
    else:
        return np.reshape(out1, newshape=[precision, precision, -1]), \
               np.reshape(out2, newshape=[precision, precision, -1]), \
               np.reshape(out3, newshape=[precision, precision, -1]), \
               np.reshape(out4, newshape=[precision, precision, -1])


def train_network_sigmoid(dataset: np.ndarray, learning_rate: float, units: int, window_size: int, log_freq=100):
    stats = RunningStats(window_size)

    network = {
        "W1": np.random.randn(2, units).astype("float32"),
        "b1": np.zeros((1, units)).astype("float32"),
        "W2": np.random.randn(units, 1).astype("float32"),
        "b2": np.zeros((1, 1)).astype("float32")
    }

    # episode = 0
    # while not stats.finished_window() or stats.window_improved():
    #     episode += 1
    #
    #     train_set, train_labels, test_set, test_labels = shuffle_dataset(dataset)
    #
    #     _, train_out2, _, train_predictions = forward(train_set, network, Dataset.CIRCLE, squeeze=True)
    #     d = -(np.subtract(train_labels, train_predictions))
    #     network["W1"] -= learning_rate * np.dot(np.multiply(d, np.multiply(train_out2.T > 0, network["W2"])), train_set).T
    #     network["b1"] -= learning_rate * np.sum(np.multiply(d, np.multiply(train_out2.T > 0, network["W2"])), axis=1)
    #     network["W2"] -= learning_rate * np.reshape(np.dot(d, np.maximum(train_out2, 0)), [units, 1])
    #     network["b2"] -= learning_rate * np.reshape(np.sum(d), [1, 1])
    #
    #     _, _, _, test_predictions = forward(test_set, network, Dataset.CIRCLE)
    #     correct = len(np.where((test_predictions[:, 0] > 0.5) == (test_labels == 1))[0])
    #     test_accuracy = 0 if correct == 0 else (correct / len(test_labels)) * 100
    #     stats.insert(test_accuracy)
    #
    #     if log_freq and episode % log_freq:
    #         print(f"Episode: {episode}, Test Accuracy: {test_accuracy:6.2f}, Running Avg: {stats.get_average():6.3f}")

    return network


def train_network_softmax(dataset: np.ndarray, learning_rate: float, units: int, window_size: int, log=True):
    stats = RunningStats(window_size)

    reg = 1e-3
    network = {
        "W1": np.random.randn(2, units).astype("float32"),
        "b1": np.zeros((1, units)).astype("float32"),
        "W2": np.random.randn(units, 2).astype("float32"),
        "b2": np.zeros((1, 2)).astype("float32")
    }

    num_examples = int(dataset.shape[0]*0.9)
    episode = 0
    while not stats.finished_window() or stats.window_improved():
        episode += 1

        train_set, train_labels, test_set, test_labels = shuffle_dataset(dataset)

        _, train_out2, _, train_predictions = forward(train_set, network, Dataset.SPIRAL, squeeze=True)

        dscores = train_predictions
        dscores[range(num_examples), train_labels] -= 1
        dscores /= num_examples

        dhidden = np.dot(dscores, network["W2"].T)
        dhidden[train_out2 <= 0] = 0
        network["W1"] -= learning_rate * reg * np.dot(train_set.T, dhidden)
        network["b1"] -= learning_rate * np.sum(dhidden, axis=0, keepdims=True)
        network["W2"] -= learning_rate * reg * np.dot(train_out2.T, dscores)
        network["b2"] -= learning_rate * np.sum(dscores, axis=0, keepdims=True)

        _, _, _, test_predictions = forward(test_set, network, Dataset.SPIRAL)
        test_accuracy = (len(np.where((test_predictions[:, 0] > 0.5) == (test_labels == 0))[0]) / len(test_labels)) * 100
        stats.insert(test_accuracy)

        if log:
            print(f"Episode: {episode}, Test Accuracy: {test_accuracy:6.2f}, Running Avg: {stats.get_average():6.3f}")

    return network


def torch_train_softmax_network(X, y, architecture, eta=3e-3, reg=1e-4, window_size=100, log=True):
    stats = RunningStats(window_size)

    classes = architecture[-1]
    layers = []
    biases = []
    for i in range(1, len(architecture)):
        layers.append(torch.randn(architecture[i-1], architecture[i], dtype=torch.float64, requires_grad=True))
        biases.append(torch.randn(1, architecture[i], dtype=torch.float64, requires_grad=True))

    optim = torch.optim.Adam(layers + biases, lr=eta)

    criterion = torch.nn.CrossEntropyLoss()

    episode = 0
    while not stats.finished_window() or stats.window_improved():
        train_set, train_labels, test_set, test_labels = dataset_to_torch(*shuffle_dataset(X, y))

        # torch forward
        train_pred = torch_forward(train_set, layers, biases, False)
        data_loss = criterion(train_pred, train_labels)

        reg_loss = 0
        for i in range(len(layers)):
            reg_loss += torch.sum(layers[i] * layers[i]) * (1 / len(layers)) * reg
        loss = data_loss + reg_loss

        loss.backward()
        test_pred = torch_forward(train_set, layers, biases, True)
        test_accuracy = (test_pred[test_labels] > 1/classes).sum() / len(test_labels) * 100
        stats.insert(test_accuracy)

        if log:
            print(f"Episode: {episode}, Test Accuracy: {test_accuracy:6.2f}, Running Avg: {stats.get_average():6.3f}")

        with torch.no_grad():
            optim.step()
            optim.zero_grad()


def save_network(network: dict, dataset: Dataset, path: str):
    units = network["W1"].shape[1]
    if not os.path.isdir(f"{path}/dataset_{dataset.name}"):
        os.mkdir(f"{path}/dataset_{dataset.name}")
    os.mkdir(f"{path}/dataset_{dataset.name}/units_{units}")
    for key in network.keys():
        with open(f"{path}/dataset_{dataset.name}/units_{units}/{key}", "wb+") as file:
            file.write(network[key].tobytes())


def check_saved_network(units: int, dataset: Dataset, path: str):
    return os.path.isdir(f"{path}/dataset_{dataset.name}/units_{units}")


def load_network(units: int, dataset: Dataset, path: str, dtype="float32"):
    classes = 1 if dataset is Dataset.CIRCLE else 2
    network = {"W1": [2, units], "b1": [1, units], "W2": [units, classes], "b2": [1, classes]}
    for key in network.keys():
        with open(f"{path}/dataset_{dataset.name}/units_{units}/{key}", "rb") as file:
            data = file.read(len(np.zeros(shape=network[key], dtype=dtype).tobytes()))
            network[key] = np.reshape(np.frombuffer(data, dtype=dtype), network[key])
            network[key].setflags(write=True)

    return network
