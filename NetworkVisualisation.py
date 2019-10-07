import numpy as np
import matplotlib.pyplot as plt
from PlotUtils import plot_to_grid
from Plot import Plot
from Plot3D import Plot3D
from matplotlib.widgets import Slider, CheckButtons
from networks.network import Network


def scale(vector, axis=0, scale=1):
    result = vector - np.min(vector, axis=axis)
    result /= np.max(result, axis=0)
    result *= scale
    return result


def setup_data_space(min_range, max_range, precision):
    dims = 2
    dim_data = [np.linspace(min_range, max_range, precision) for _ in range(dims)]
    data_grid = np.meshgrid(*dim_data)
    return np.stack([data_dim.flatten() for data_dim in data_grid], axis=1), dim_data


def scale_out2(out2, w2, all_p1_enabled, perceptron2, precision):
    reshaped_out2 = np.reshape(out2, [precision * precision, -1])
    scaled_outs = np.stack([reshaped_out2[:, i] * w2[perceptron, perceptron2] for i, perceptron in enumerate(all_p1_enabled)], axis=1)
    return np.reshape(scaled_outs, [precision, precision, -1])


class NetworkVisualisation(object):

    def __init__(self, X: np.ndarray, y: np.ndarray, network: Network, min_range: int, max_range: int, quality: int, saves_path: str = None, seed: int = 1):
        np.random.seed(seed)

        self.precision = quality
        self.min_range = min_range
        self.max_range = max_range
        self.data_points = X
        self.data_labels = y
        self.data_space, self.dim_data = setup_data_space(min_range, max_range, quality)
        self.network = network
        self.original_layers = [network.to_np(layer) for layer in network.layers]
        self.original_biases = [network.to_np(bias) for bias in network.biases]

        # Network Creation
        # if saves_path and check_saved_network(units, dataset, saves_path):
        #     self.network = load_network(units, dataset, saves_path)
        # else:
        #     if dataset is Dataset.CIRCLE:
        #         self.network = train_network_sigmoid(self.dataset, units=units, learning_rate=5e-3, window_size=1000)
        #     elif dataset is Dataset.SPIRAL:
        #         self.network = train_network_softmax(self.dataset, units=units, learning_rate=1, window_size=1000)
        #     else:
        #         raise Exception("Invalid dataset type")
        #     if saves_path:
        #         save_network(self.network, dataset, saves_path)

        # GUI Visualisation
        self.neuron1 = 0
        self.is_relu = True
        self.neuron2 = 0
        self.p2c = 0
        self.is_pre_add = False
        self.is_sig = True
        self.selected_architecture = [set(range(layer_n)) for layer_n in self.network.architecture]
        self.ignore_update = False
        self.cut = False

        fig = plt.figure(figsize=(13, 6.5))
        self.plot_network(fig)
        self.plot_controls(fig)

    def plot_network(self, fig):
        results = [self.network.to_np(result) for result in self.network.forward(self.network.from_np(self.data_space), precision=self.precision, is_mlp=self.cut)]
        grouped_points = [self.data_points[self.data_labels == i] for i in range(self.network.n_classes)]

        colours = ["r", "g", "b"]

        self.layer1_plot = Plot(fig, (4, 4), (0, 0), (1, 3), results[2][:, :, 0], self.min_range, self.max_range)
        for i in range(self.network.n_classes):
            self.layer1_plot.ax.scatter(grouped_points[i][:, 0], grouped_points[i][:, 1], s=3, c=colours[i], alpha=0.5)

        self.layer1_3d_plot = Plot3D(fig, (4, 4), (1, 0), (1, 3), self.precision, results[2])

        self.layer2_plot = Plot(fig, (4, 4), (2, 0), (1, 3), results[-1][:, :, 0], self.min_range, self.max_range)
        for i in range(self.network.n_classes):
            self.layer2_plot.ax.scatter(grouped_points[i][:, 0], grouped_points[i][:, 1], s=3, c=colours[i], alpha=0.5)

        self.layer2_3d_plot = Plot3D(fig, (4, 4), (3, 0), (1, 3), self.precision, results[-1])

    def plot_controls(self, fig):
        step_size = 0.01
        padding = 5

        layer_1 = self.network.to_np(self.network.layers[0])
        bias_1 = self.network.to_np(self.network.biases[0])

        # Plot 1 controls
        w1x_min = layer_1[0].min()
        w1x_max = layer_1[0].max()
        w1x_diff = (w1x_max - w1x_min) / 2 + padding

        w1y_min = layer_1[1].min()
        w1y_max = layer_1[1].max()
        w1y_diff = (w1y_max - w1y_min) / 2 + padding

        w1b_min = bias_1.min()
        w1b_max = bias_1.max()
        w1b_diff = (w1b_max - w1b_min) / 2 + padding

        p1x_ax = plot_to_grid(fig, (2, 16), (0, 12), (1, 1))
        self.p1x_slid = Slider(p1x_ax, 'x', valmin=w1x_min - w1x_diff, valmax=w1x_max + w1x_diff, valinit=layer_1[0, 0], valstep=step_size)
        self.p1x_slid.on_changed(self.p1x_changed)

        p1y_ax = plot_to_grid(fig, (2, 16), (0, 13), (1, 1))
        self.p1y_slid = Slider(p1y_ax, 'y', valmin=w1y_min - w1y_diff, valmax=w1y_max + w1y_diff, valinit=layer_1[1, 0], valstep=step_size)
        self.p1y_slid.on_changed(self.p1y_changed)

        p1b_ax = plot_to_grid(fig, (24, 16), (0, 14), (7, 1))
        self.p1b_slid = Slider(p1b_ax, 'b', valmin=w1b_min - w1b_diff, valmax=w1b_max + w1b_diff, valinit=bias_1[0, 0], valstep=step_size)
        self.p1b_slid.on_changed(self.p1b_changed)

        p1_ax = plot_to_grid(fig, (24, 16), (0, 15), (7, 1))
        self.p1_slid = Slider(p1_ax, 'n', valmin=0, valmax=layer_1.shape[1] - 1, valinit=self.neuron1, valstep=1)
        self.p1_slid.on_changed(self.p1_changed)

        p1_opt_ax = plot_to_grid(fig, (24, 16), (8, 14), (3, 3))
        self.p1_opt_buttons = CheckButtons(p1_opt_ax, ["ReLU?", "Cut?", "Enabled?"], [self.is_relu, self.cut, True])
        self.p1_opt_buttons.on_clicked(self.p1_options_update)

        # Plot 2 Controls
        layer_2 = self.network.to_np(self.network.layers[-1])
        bias_2 = self.network.to_np(self.network.biases[-1])

        w2_min = layer_2.min()
        w2_max = layer_2.max()
        w2_diff = (w2_max - w2_min) / 2 + padding

        w2b_abs = np.abs(bias_2[0, 0]) + padding
        w2b_min = bias_2[0, 0] - w2b_abs
        w2b_max = bias_2[0, 0] + w2b_abs

        p2_weight_val_ax = plot_to_grid(fig, (2, 16), (1, 12), (1, 1))
        self.p2w_slid = Slider(p2_weight_val_ax, 'w', valmin=w2_min - w2_diff, valmax=w2_max + w2_diff, valinit=layer_2[0, 0], valstep=step_size)
        self.p2w_slid.on_changed(self.p2w_changed)

        p2_connection_dim_ax = plot_to_grid(fig, (2, 16), (1, 13), (1, 1))
        self.p2c_slid = Slider(p2_connection_dim_ax, 'c', valmin=0, valmax=layer_2.shape[0] - 1, valinit=0, valstep=1)
        self.p2c_slid.on_changed(self.p2c_changed)

        p2b_ax = plot_to_grid(fig, (24, 16), (13, 14), (7, 1))
        self.p2b_slid = Slider(p2b_ax, 'b', valmin=w2b_min, valmax=w2b_max, valinit=bias_2[0, 0], valstep=step_size)
        self.p2b_slid.on_changed(self.p2b_changed)

        p2_ax = plot_to_grid(fig, (24, 16), (13, 15), (7, 1))
        self.p2_slid = Slider(p2_ax, 'n', valmin=0, valmax=layer_2.shape[1] - 1, valinit=self.neuron2, valstep=1)
        self.p2_slid.on_changed(self.p2_changed)

        p2_opt_ax = plot_to_grid(fig, (24, 16), (21, 14), (4, 2))
        self.p2_opt_buttons = CheckButtons(p2_opt_ax, ["Pre-add?", "Transform?", "Enabled?"], [self.is_pre_add, self.is_sig, True])
        self.p2_opt_buttons.on_clicked(self.p2_options_update)

    def p1_changed(self, val):
        if not self.ignore_update:
            self.neuron1 = int(val)
            self.ignore_update = True
            self.update_plot1_widgets()
            self.ignore_update = False

            self.update_just_plot1()

    def p1x_changed(self, val):
        if not self.ignore_update:
            self.network.layers[0][0, self.neuron1] = val
            self.update_visuals()

    def p1y_changed(self, val):
        if not self.ignore_update:
            self.network.layers[0][1, self.neuron1] = val
            self.update_visuals()

    def p1b_changed(self, val):
        if not self.ignore_update:
            self.network.biases[0][0, self.neuron1] = val
            self.update_visuals()

    def p1_options_update(self, label):
        if not self.ignore_update:
            if label == "ReLU?":
                self.is_relu = not self.is_relu
                self.update_just_plot1()
            elif label == "Enabled?":
                is_enabled = self.p1_opt_buttons.get_status()[2]
                if is_enabled and self.neuron1 not in self.selected_architecture[1]:
                    self.selected_architecture[1].add(self.neuron1)
                elif not is_enabled and self.neuron1 in self.selected_architecture[1]:
                    layer1_out = sorted(list(self.selected_architecture[1])).index(self.neuron1)
                    self.layer1_3d_plot.remove_plot(layer1_out)
                    self.selected_architecture[1].remove(self.neuron1)
            elif label == "Cut?":
                self.cut = not self.cut

            self.update_visuals()

    def p2w_changed(self, val):
        if not self.ignore_update:
            self.network.layers[-1][self.p2c, 0] = val

            self.update_just_plot2()

    def p2c_changed(self, val):
        if not self.ignore_update:
            self.p2c = int(val)
            self.ignore_update = True
            self.p2w_slid.set_val(self.network.layers[-1][self.p2c, self.neuron2])
            self.p2w_slid.vline.set_xdata(self.original_layers[-1][self.p2c, self.neuron2])
            self.ignore_update = False

    def p2b_changed(self, val):
        if not self.ignore_update:
            self.network.biases[-1][0, 0] = val
            self.update_just_plot2()

    def p2_changed(self, val):
        if not self.ignore_update:
            self.neuron2 = int(val)
            self.ignore_update = True
            self.update_plot2_widgets()
            self.ignore_update = False

            self.update_just_plot2()

    def p2_options_update(self, label):
        if not self.ignore_update:
            if label == "Transform?":
                self.is_sig = not self.is_sig
            elif label == "Pre-add?":
                self.is_pre_add = not self.is_pre_add
            elif label == "Enabled?":
                is_enabled = self.p2_opt_buttons.get_status()[2]
                if is_enabled and self.neuron2 not in self.selected_architecture[-1]:
                    self.selected_architecture[-1].add(self.neuron2)
                elif not is_enabled and self.neuron2 in self.selected_architecture[-1]:
                    layer2_out = sorted(list(self.selected_architecture[-1])).index(self.neuron2)
                    self.layer2_3d_plot.remove_plot(layer2_out)
                    self.selected_architecture[-1].remove(self.neuron2)
            self.update_just_plot2()

    def show(self):
        plt.show()

    def update_plot1(self, out1, out2):
        if not self.ignore_update:
            if self.neuron1 in self.selected_architecture[1]:
                self.layer1_plot.set_visible(True)
                layer1_out = sorted(list(self.selected_architecture[1])).index(self.neuron1)
                if not self.is_relu:
                    layer1_data = out1[:, :, layer1_out]
                else:
                    layer1_data = out2[:, :, layer1_out]
                self.layer1_plot.update(layer1_data)
            else:
                self.layer1_plot.set_visible(False)

    def update_3d_plot1(self, out1, out2):
        if not self.ignore_update:
            if self.neuron1 in self.selected_architecture[1]:
                if not self.is_relu:
                    self.layer1_3d_plot.update_all(out1)
                else:
                    self.layer1_3d_plot.update_all(out2)

    def update_plot2(self, out2, out3, out4):
        if not self.ignore_update:
            if self.neuron2 in self.selected_architecture[-1]:
                self.layer2_plot.set_visible(True)
                neuron2_index = sorted(list(self.selected_architecture[-1])).index(self.neuron2)
                if self.is_pre_add:
                    layer2_data = scale_out2(out2, self.network.to_np(self.network.layers[-1]), self.selected_architecture[1], neuron2_index, self.precision)
                    layer2_data = np.sum(layer2_data, axis=2)
                elif not self.is_sig:
                    layer2_data = out3[:, :, neuron2_index]
                else:
                    layer2_data = out4[:, :, neuron2_index]
                self.layer2_plot.update(layer2_data)
            else:
                self.layer2_plot.set_visible(False)

    def update_3d_plot2(self, out2, out3, out4):
        if not self.ignore_update:
            if self.is_pre_add:
                layer2_data = scale_out2(out2, self.network.to_np(self.network.layers[-1]), self.selected_architecture[1], self.neuron2, self.precision)
            elif not self.is_sig:
                layer2_data = out3
            else:
                layer2_data = out4
            self.layer2_3d_plot.update_all(layer2_data)

    def update_visuals(self):
        if not self.ignore_update:
            results = [self.network.to_np(result) for result in self.network.forward(self.network.from_np(self.data_space), arch_selection=self.selected_architecture, precision=self.precision, is_mlp=self.cut)]
            self.update_plot1_visuals(results[1], results[2])
            self.update_plot2_visuals(results[-3], results[-2], results[-1])
            plt.draw()

    def update_just_plot1(self):
        if not self.ignore_update:
            results = [self.network.to_np(result) for result in self.network.forward(self.network.from_np(self.data_space), arch_selection=self.selected_architecture, precision=self.precision, is_mlp=self.cut)]
            self.update_plot1_visuals(results[1], results[2])
            plt.draw()

    def update_plot1_visuals(self, out1, out2):
        if not self.ignore_update:
            self.update_plot1(out1, out2)
            self.update_3d_plot1(out1, out2)

    def update_just_plot2(self):
        if not self.ignore_update:
            results = [self.network.to_np(result) for result in self.network.forward(self.network.from_np(self.data_space), arch_selection=self.selected_architecture, precision=self.precision, is_mlp=self.cut)]
            self.update_plot2_visuals(results[-3], results[-2], results[-1])
            plt.draw()

    def update_plot2_visuals(self, out2, out3, out4):
        if not self.ignore_update:
            self.update_plot2(out2, out3, out4)
            self.update_3d_plot2(out2, out3, out4)

    def update_plot1_widgets(self):
        self.p1b_slid.set_val(self.network.to_np(self.network.biases[0])[0, self.neuron1])
        self.p1x_slid.set_val(self.network.to_np(self.network.layers[0])[0, self.neuron1])
        self.p1y_slid.set_val(self.network.to_np(self.network.layers[0])[1, self.neuron1])

        self.p1b_slid.vline.set_xdata(self.original_biases[0][0, self.neuron1])
        self.p1x_slid.vline.set_xdata(self.original_layers[0][0, self.neuron1])
        self.p1y_slid.vline.set_xdata(self.original_layers[0][1, self.neuron1])

        if (self.neuron1 in self.selected_architecture[1] and not self.p1_opt_buttons.get_status()[2]) or \
                (self.neuron1 not in self.selected_architecture[1] and self.p1_opt_buttons.get_status()[2]):
            self.p1_opt_buttons.set_active(2)

    def update_plot2_widgets(self):
        self.p2b_slid.set_val(self.network.to_np(self.network.biases[-1])[0, self.neuron2])
        self.p2w_slid.set_val(self.network.to_np(self.network.layers[-1])[self.p2c, self.neuron2])

        self.p2b_slid.vline.set_xdata(self.original_biases[-1][0, self.neuron2])
        self.p2w_slid.vline.set_xdata(self.original_layers[-1][self.p2c, self.neuron2])

        if (self.neuron2 in self.selected_architecture[-1] and not self.p2_opt_buttons.get_status()[2]) or \
                (self.neuron2 not in self.selected_architecture[-1] and self.p2_opt_buttons.get_status()[2]):
            self.p2_opt_buttons.set_active(2)

    def update_widgets(self):
        self.update_plot1_widgets()
        self.update_plot2_widgets()
