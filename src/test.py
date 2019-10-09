from src.Datasets import spiral_dataset

from src.networks.torch_network import TorchNetwork

from src.NetworkVisualisation import NetworkVisualisation

if __name__ == '__main__':
    save_path = "/home/michael/dev/network-visualisation/resources/saves/test3"
    X, y = spiral_dataset(1000, 3)
    network_class = TorchNetwork
    # X, y = circle_dataset(1000, -1, 1, 0.8)

    if not network_class.save_exists(save_path):
        architecture = [2, 12, 6, 3]
        lr = 5e-1
        reg = 1e-4
        network = network_class(architecture, lr, reg)
        network.train(X, y, 200)
        network.save(save_path)
    else:
        network = network_class.load(save_path)
    NetworkVisualisation(X, y, network, -1, 1, 50).show()
