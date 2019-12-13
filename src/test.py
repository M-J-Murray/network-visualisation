from src.Datasets import complex_circles_dataset, perfume_dataset
from itertools import combinations_with_replacement
import numpy as np

from src.networks.torch_network import TorchNetwork

from src.NetworkVisualisation import NetworkVisualisation



def test_architectures(in_dim, out_dim, tests, min_width, max_width, steps, max_depth):
    lr = 1e-3
    reg = 1e-4
    all_arch_scores = []
    best_arch = None
    best_acc = None
    for d in range(1, max_depth):
        comb = combinations_with_replacement(np.linspace(min_width, max_width, steps), d)
        for sample in list(comb):
            arch = [in_dim] + [int(s) for s in sample] + [out_dim]
            print("Testing arch: " + str(arch), end='')
            avg_acc = 0
            for i in range(tests):
                net = TorchNetwork(arch, lr, reg)
                avg_acc += net.train(X, y, 100, log=False)
            avg_acc /= tests
            all_arch_scores.append((arch, avg_acc))
            if best_acc is None or avg_acc > best_acc:
                best_acc = avg_acc
                best_arch = arch
                print(" - New Best")
            else:
                print()
    print("Testing finished, best arch: " + str(best_arch))
    print("Final rankings")
    all_arch_scores.sort(key=lambda x: x[1], reverse=True)
    for arch, score in all_arch_scores:
        print(str(arch)+", "+str(score))


def test_predefined_architectures(architectures, tests):
    lr = 1e-3
    reg = 1e-4
    all_arch_scores = []
    best_arch = None
    best_acc = None
    for arch in list(architectures):
        print("Testing arch: " + str(arch), end='')
        avg_acc = 0
        for i in range(tests):
            net = TorchNetwork(arch, lr, reg)
            avg_acc += net.train(X, y, 100, log=False)
        avg_acc /= tests
        all_arch_scores.append((arch, avg_acc))
        if best_acc is None or avg_acc > best_acc:
            best_acc = avg_acc
            best_arch = arch
            print(" - New Best")
        else:
            print()
    print("Testing finished, best arch: " + str(best_arch))
    print("Final rankings")
    all_arch_scores.sort(key=lambda x: x[1], reverse=True)
    for arch, score in all_arch_scores:
        print(str(arch) + ", " + str(score))


if __name__ == '__main__':
    save_path = "/home/michael/dev/network-visualisation/resources/saves/test"
    # X, y = spiral_dataset(1000, 3)
    network_class = TorchNetwork
    X, y = complex_circles_dataset(2000, -1, 1, 0.5, 0.3)
    # X, y = perfume_dataset("resources/datasets/perfume_data_pairs.csv")

    # test_architectures(2, 3, 5, 6, 36, 6, 6)
    # test_architectures(2, 3, 1, 6, 36, 2, 2)

    # architectures = [
    #     [2, 30, 30, 30, 30, 3],
    #     [2, 36, 36, 36, 36, 3],
    #     [2, 30, 36, 36, 3],
    #     [2, 24, 36, 36, 36, 36, 3],
    #     [2, 30, 36, 36, 36, 3],
    #     [2, 18, 24, 30, 30, 3],
    #     [2, 18, 24, 24, 36, 36, 3],
    #     [2, 30, 30, 30, 36, 3],
    #     [2, 30, 30, 36, 36, 36, 3],
    #     [2, 24, 30, 36, 36, 3],
    #     [2, 18, 24, 36, 36, 36, 3]
    # ]
    # test_predefined_architectures(reversed(architectures), 10)

    if not network_class.save_exists(save_path):
        architecture = [2, 30, 30, 36, 36, 36, 3]
        lr = 1e-3
        reg = 1e-4
        network = network_class(architecture, lr, reg)
        accuracy = network.train(X, y, 100)
        network.save(save_path)
    else:
        network = network_class.load(save_path)
    NetworkVisualisation(X, y, network, 100).show()
