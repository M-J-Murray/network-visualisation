import numpy as np
from src.PlotUtils import subplot_3d


class Plot3D(object):

    def __init__(self, fig, plot, loc, shape, precision, data):
        self.precision = precision
        self.plots, self.axes = subplot_3d(fig, plot, loc, shape, precision, data)

    def update_plane(self, all_data, plane, index):
        if self.plots[index]:
            for cont in self.plots[index].collections:
                cont.remove()
        vmax = np.max(all_data)
        vmin = np.min(all_data)
        self.plots[index] = self.axes.contourf(range(self.precision), range(self.precision), plane, self.precision, vmin=vmin, vmax=vmax, cmap='plasma')

    def update_all(self, all_data):
        self.remove_all_plots()
        vmax = np.max(all_data)
        vmin = np.min(all_data)
        self.plots = [None]*all_data.shape[2]
        for i in range(all_data.shape[2]):
            self.plots[i] = self.axes.contourf(range(self.precision), range(self.precision), all_data[:, :, i], self.precision, vmin=vmin, vmax=vmax, cmap='plasma')

    def remove_all_plots(self):
        for i in range(len(self.plots)):
            self.remove_plot(i)

    def remove_plot(self, index):
        if self.plots[index]:
            for cont in self.plots[index].collections:
                cont.remove()
            self.plots[index] = None
