from Datasets import circle_dataset, spiral_dataset, nested_circle_dataset, complex_circles_dataset

from networks.torch_network import TorchNetwork
from networks.numpy_network import NumpyNetwork

from NetworkVisualisation import NetworkVisualisation

if __name__ == '__main__':
    # NetworkVisualisation(units=1, data_points=6, min_range=-1, max_range=1, quality=100, saves_path="resources/Saves", dataset=Dataset.CIRCLE).show()
    # NetworkVisualisation(units=4, data_points=1000, min_range=-1, max_range=1, quality=100, dataset=Dataset.CIRCLE).show()
    # NetworkVisualisation(units=24, data_points=1000, min_range=-1, max_range=1, quality=100, saves_path="resources/Saves", dataset=Dataset.SPIRAL).show()
    save_path = "/home/michael/dev/network-visualisation/resources/Saves/test1"
    X, y = spiral_dataset(1000, 3)
    network_class = TorchNetwork
    # X, y = circle_dataset(1000, -1, 1, 0.8)

    if not network_class.save_exists(save_path):
        architecture = [2, 12, 3]
        lr = 5e-1
        reg = 1e-4
        network = network_class(architecture, lr, reg)
        # network = TorchNetwork(architecture, lr, reg)
        network.train(X, y, 200)
        network.save(save_path)
    else:
        network = network_class.load(save_path)
    NetworkVisualisation(X, y, network, -1, 1, 50).show()
