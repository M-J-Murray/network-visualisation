import matplotlib.pyplot as plt
from PlotUtils import plot_to_grid
from Plot import Plot
from Plot3D import Plot3D
from matplotlib.widgets import Slider, CheckButtons
from Datasets import get_spiral_dataset, get_circle_dataset
from NetworkUtils import *


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

    def __init__(self, units, data_points, min_range, max_range, quality, dataset, saves_path=None, seed=1):
        np.random.seed(seed)

        self.precision = quality
        self.min_range = min_range
        self.max_range = max_range
        self.dataset_type = dataset
        if dataset is Dataset.CIRCLE:
            self.dataset = get_circle_dataset(points=data_points, min_range=min_range, max_range=max_range, radius=0.8)
        elif dataset is Dataset.SPIRAL:
            self.dataset = get_spiral_dataset(data_points, classes=2)
        else:
            raise Exception("Invalid dataset type")
        self.data_points = self.dataset[:, :-1]
        self.data_labels = self.dataset[:, -1]
        self.data_space, self.dim_data = setup_data_space(min_range, max_range, quality)

        # Network Creation
        if saves_path and check_saved_network(units, dataset, saves_path):
            self.network = load_network(units, dataset, saves_path)
        else:
            if dataset is Dataset.CIRCLE:
                self.network = train_network_sigmoid(self.dataset, units=units, learning_rate=5e-3, window_size=1000)
            elif dataset is Dataset.SPIRAL:
                self.network = train_network_softmax(self.dataset, units=units, learning_rate=1, window_size=1000)
            else:
                raise Exception("Invalid dataset type")
            save_network(self.network, dataset, saves_path)
        self.default_network = dict(zip(self.network.keys(), [layer.copy() for layer in self.network.values()]))

        # GUI Visualisation
        self.perceptron1 = 0
        self.is_relu = True
        self.perceptron2 = 0
        self.connection = 0
        self.is_pre_add = False
        self.is_sig = True
        self.all_p1_enabled = set(range(self.network["W1"].shape[1]))
        self.ignore_update = False

        fig = plt.figure(figsize=(13, 6.5))
        self.plot_network(fig)
        self.plot_controls(fig)

    def plot_network(self, fig):
        _, out2, out3, out4 = forward(self.data_space, self.network, self.dataset_type, precision=self.precision)
        outer_points = self.data_points[self.data_labels == 1]
        inner_points = self.data_points[self.data_labels == 0]

        self.layer1_plot = Plot(fig, (4, 4), (0, 0), (1, 3), out2[:, :, 0], self.min_range, self.max_range)
        self.layer1_plot.ax.scatter(outer_points[:, 0], outer_points[:, 1], s=3, c="g", alpha=0.5)
        self.layer1_plot.ax.scatter(inner_points[:, 0], inner_points[:, 1], s=3, c="r", alpha=0.5)

        self.layer1_3d_plot = Plot3D(fig, (4, 4), (1, 0), (1, 3), self.precision, out2)

        self.layer2_plot = Plot(fig, (4, 4), (2, 0), (1, 3), out4[:, :, 0], self.min_range, self.max_range)
        self.layer2_plot.ax.scatter(outer_points[:, 0], outer_points[:, 1], s=3, c="g", alpha=0.5)
        self.layer2_plot.ax.scatter(inner_points[:, 0], inner_points[:, 1], s=3, c="r", alpha=0.5)

        self.layer2_3d_plot = Plot3D(fig, (4, 4), (3, 0), (1, 3), self.precision, out4)

    def plot_controls(self, fig):
        step_size = 0.01
        padding = 5

        # Plot 1 controls
        w1x_min = self.network["W1"][0].min()
        w1x_max = self.network["W1"][0].max()
        w1x_diff = (w1x_max - w1x_min)/2 + padding

        w1y_min = self.network["W1"][1].min()
        w1y_max = self.network["W1"][1].max()
        w1y_diff = (w1y_max - w1y_min) / 2 + padding

        w1b_min = self.network["b1"].min()
        w1b_max = self.network["b1"].max()
        w1b_diff = (w1b_max - w1b_min) / 2 + padding

        p1x_ax = plot_to_grid(fig, (2, 16), (0, 12), (1, 1))
        self.p1x_slid = Slider(p1x_ax, 'P1 x', valmin=w1x_min-w1x_diff, valmax=w1x_max+w1x_diff,  valinit=self.network["W1"][0, 0], valstep=step_size)
        self.p1x_slid.on_changed(self.p1x_changed)

        p1y_ax = plot_to_grid(fig, (2, 16), (0, 13), (1, 1))
        self.p1y_slid = Slider(p1y_ax, 'P1 y', valmin=w1y_min-w1y_diff, valmax=w1y_max+w1y_diff, valinit=self.network["W1"][1, 0], valstep=step_size)
        self.p1y_slid.on_changed(self.p1y_changed)

        p1b_ax = plot_to_grid(fig, (24, 16), (0, 14), (7, 1))
        self.p1b_slid = Slider(p1b_ax, 'P1 b', valmin=w1b_min-w1b_diff, valmax=w1b_max+w1b_diff, valinit=self.network["b1"][0, 0], valstep=step_size)
        self.p1b_slid.on_changed(self.p1b_changed)

        p1_ax = plot_to_grid(fig, (24, 16), (0, 15), (7, 1))
        self.p1_slid = Slider(p1_ax, 'P1', valmin=0, valmax=self.network["W1"].shape[1]-1, valinit=self.perceptron1, valstep=1)
        self.p1_slid.on_changed(self.p1_changed)

        p1_opt_ax = plot_to_grid(fig, (24, 16), (8, 14), (3, 2))
        self.p1_opt_buttons = CheckButtons(p1_opt_ax, ["ReLU?", "Enabled?"], [self.is_relu, True])
        self.p1_opt_buttons.on_clicked(self.p1_options_update)


        # Plot 2 Controls
        w2_min = self.network["W2"].min()
        w2_max = self.network["W2"].max()
        w2_diff = (w2_max - w2_min) / 2 + padding

        w2b_abs = np.abs(self.network["b2"][0, 0]) + padding
        w2b_min = self.network["b2"][0, 0]-w2b_abs
        w2b_max = self.network["b2"][0, 0]+w2b_abs

        p2_weight_val_ax = plot_to_grid(fig, (2, 16), (1, 12), (1, 1))
        self.p2_dim_val_slid = Slider(p2_weight_val_ax, 'p2 w', valmin=w2_min-w2_diff, valmax=w2_max+w2_diff, valinit=self.network["W2"][0, 0], valstep=step_size)
        self.p2_dim_val_slid.on_changed(self.p2_weight_changed)

        p2_connection_dim_ax = plot_to_grid(fig, (2, 16), (1, 13), (1, 1))
        self.p2_connection_dim_slid = Slider(p2_connection_dim_ax, 'p2 c', valmin=0, valmax=self.network["W2"].shape[0]-1, valinit=0, valstep=1)
        self.p2_connection_dim_slid.on_changed(self.p2_connection_dim_changed)

        p2b_ax = plot_to_grid(fig, (24, 16), (13, 14), (7, 1))
        self.p2b_slid = Slider(p2b_ax, 'p2 b', valmin=w2b_min, valmax=w2b_max, valinit=self.network["b2"][0, 0], valstep=step_size)
        self.p2b_slid.on_changed(self.p2b_changed)

        p2_opt_ax = plot_to_grid(fig, (24, 16), (21, 14), (4, 2))
        self.p2_opt_buttons = CheckButtons(p2_opt_ax, ["Pre-add?", "Transform?"], [self.is_pre_add, self.is_sig])
        self.p2_opt_buttons.on_clicked(self.p2_options_update)

    def p1_changed(self, val):
        self.perceptron1 = int(val)
        self.ignore_update = True
        self.update_widgets()
        self.ignore_update = False

        self.update_just_plot1()

    def p1x_changed(self, val):
        self.network["W1"][0, self.perceptron1] = val
        self.update_visuals()

    def p1y_changed(self, val):
        self.network["W1"][1, self.perceptron1] = val
        self.update_visuals()

    def p1b_changed(self, val):
        self.network["b1"][0, self.perceptron1] = val
        self.update_visuals()

    def p1_options_update(self, label):
        if label == "ReLU?":
            self.is_relu = not self.is_relu
            self.update_just_plot1()
        elif label == "Enabled?":
            is_enabled = self.p1_opt_buttons.get_status()[1]
            if is_enabled and self.perceptron1 not in self.all_p1_enabled:
                self.all_p1_enabled.add(self.perceptron1)
            elif not is_enabled and self.perceptron1 in self.all_p1_enabled:
                layer1_out = sorted(list(self.all_p1_enabled)).index(self.perceptron1)
                self.layer1_3d_plot.remove_plot(layer1_out)
                self.all_p1_enabled.remove(self.perceptron1)

            self.update_visuals()

    def p2_weight_changed(self, val):
        self.network["W2"][self.connection, 0] = val

        self.update_just_plot2()

    def p2_connection_dim_changed(self, val):
        self.connection = int(val)
        self.ignore_update = True
        self.p2_dim_val_slid.set_val(self.network["W2"][self.connection, 0])
        self.p2_dim_val_slid.vline.set_xdata(self.default_network["W2"][self.connection, 0])
        self.ignore_update = False

    def p2b_changed(self, val):
        self.network["b2"][0, 0] = val
        self.update_just_plot2()

    def p2_options_update(self, label):
        if label == "Transform?":
            self.is_sig = not self.is_sig
        elif label == "Pre-add?":
            self.is_pre_add = not self.is_pre_add

        self.update_just_plot2()

    def show(self):
        plt.show()

    def update_plot1(self, out1, out2):
        if self.perceptron1 in self.all_p1_enabled:
            self.layer1_plot.set_visible(True)
            layer1_out = sorted(list(self.all_p1_enabled)).index(self.perceptron1)
            if not self.is_relu:
                layer1_data = out1[:, :, layer1_out]
            else:
                layer1_data = out2[:, :, layer1_out]
            self.layer1_plot.update(layer1_data)
        else:
            self.layer1_plot.set_visible(False)

    def update_3d_plot1(self, out1, out2):
        if self.perceptron1 in self.all_p1_enabled:
            if not self.is_relu:
                self.layer1_3d_plot.update_all(out1)
            else:
                self.layer1_3d_plot.update_all(out2)

    def update_plot2(self, out2, out3, out4):
        if self.is_pre_add:
            layer2_data = scale_out2(out2, self.network["W2"], self.all_p1_enabled, self.perceptron2, self.precision)
            layer2_data = np.sum(layer2_data, axis=2)
        elif not self.is_sig:
            layer2_data = out3[:, :, 0]
        else:
            layer2_data = out4[:, :, 0]
        self.layer2_plot.update(layer2_data)

    def update_3d_plot2(self, out2, out3, out4):
        if self.is_pre_add:
            layer2_data = scale_out2(out2, self.network["W2"], self.all_p1_enabled, self.perceptron2, self.precision)
        elif not self.is_sig:
            layer2_data = out3
        else:
            layer2_data = out4
        self.layer2_3d_plot.update_all(layer2_data)

    def update_visuals(self):
        if not self.ignore_update:
            out1, out2, out3, out4 = forward(self.data_space, self.network, self.dataset_type, self.all_p1_enabled, self.precision)
            self.update_plot1_visuals(out1, out2)
            self.update_plot2_visuals(out2, out3, out4)
            plt.draw()

    def update_just_plot1(self):
        if not self.ignore_update:
            out1, out2, out3, out4 = forward(self.data_space, self.network, self.dataset_type, self.all_p1_enabled, self.precision)
            self.update_plot1_visuals(out1, out2)
            plt.draw()

    def update_plot1_visuals(self, out1, out2):
        self.update_plot1(out1, out2)
        self.update_3d_plot1(out1, out2)

    def update_just_plot2(self):
        if not self.ignore_update:
            out1, out2, out3, out4 = forward(self.data_space, self.network, self.dataset_type, self.all_p1_enabled, self.precision)
            self.update_plot2_visuals(out2, out3, out4)
            plt.draw()

    def update_plot2_visuals(self, out2, out3, out4):
        self.update_plot2(out2, out3, out4)
        self.update_3d_plot2(out2, out3, out4)

    def update_widgets(self):
        self.p1b_slid.set_val(self.network["b1"][0, self.perceptron1])
        self.p1x_slid.set_val(self.network["W1"][0, self.perceptron1])
        self.p1y_slid.set_val(self.network["W1"][1, self.perceptron1])

        self.p1b_slid.vline.set_xdata(self.default_network["b1"][0, self.perceptron1])
        self.p1x_slid.vline.set_xdata(self.default_network["W1"][0, self.perceptron1])
        self.p1y_slid.vline.set_xdata(self.default_network["W1"][1, self.perceptron1])

        if (self.perceptron1 in self.all_p1_enabled and not self.p1_opt_buttons.get_status()[1]) or \
           (self.perceptron1 not in self.all_p1_enabled and self.p1_opt_buttons.get_status()[1]):
            self.p1_opt_buttons.set_active(1)


if __name__ == '__main__':
    NetworkVisualisation(units=5, data_points=1000, min_range=-1, max_range=1, quality=100, saves_path="resources/Saves", dataset=Dataset.CIRCLE).show()
    # NetworkVisualisation(units=24, data_points=1000, min_range=-1, max_range=1, quality=100, saves_path="resources/Saves", dataset=Dataset.SPIRAL).show()
