import numpy as np
from PlotUtils import subplot


class Plot(object):

    def __init__(self, fig, plot, loc, shape, data, min_range, max_range):
        self.plot, self.ax = subplot(fig, plot, loc, shape, data, min_range, max_range)

    def update(self, data):
        self.plot.set_data(data)
        self.plot.set_clim([np.min(data), np.max(data)])

    def set_visible(self, is_visible):
        self.plot.set_visible(is_visible)
