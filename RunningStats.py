import numpy as np


class RunningStats(object):

    def __init__(self, window_size):
        self.index = 1
        self.window_size = window_size
        self.values = np.zeros(window_size)
        self.max_avg = 0
        self.last_window_avg = 0

    def get_average(self):
        avg = np.average(self.values)
        if avg > self.max_avg:
            self.max_avg = avg
        return avg

    def insert(self, value):
        self.values[self.index] = value
        self.index += 1
        if self.index >= self.window_size:
            self.index = 0

    def finished_window(self):
        return self.index == 0

    def window_improved(self):
        window_avg = self.get_average()
        result = window_avg > self.last_window_avg
        self.last_window_avg = window_avg
        return result
