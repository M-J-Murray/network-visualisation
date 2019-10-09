import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D # IS USED DO NOT REMOVE!!!


def plot_to_grid(fig, plot, location, shape, is_3d=False):
    gs = gridspec.GridSpec(*reversed(plot))
    if is_3d:
        return fig.add_subplot(gs[location[1]:location[1]+shape[1], location[0]:location[0]+shape[0]], projection='3d')
    else:
        return fig.add_subplot(gs[location[1]:location[1]+shape[1], location[0]:location[0]+shape[0]])


def subplot_3d(fig, plot, location, shape, precision, data):
    ax = plot_to_grid(fig, plot, location, shape, is_3d=True)
    # ax.view_init(60, 0) - sets angle
    vmax = np.max(data)
    vmin = np.min(data)
    ims = []
    for i in range(data.shape[2]):
        cnt = ax.contourf(range(precision), range(precision), data[:, :, i], precision, vmin=vmin, vmax=vmax, cmap='plasma', origin='lower')
        ims.append(cnt)
    ax.axis('off')
    return ims, ax


def subplot(fig, plot, location, shape, data, min_range, max_range):
    ax = plot_to_grid(fig, plot, location, shape)
    im = ax.imshow(data, interpolation='nearest', cmap='plasma', extent=[min_range, max_range, min_range, max_range], origin='lower')
    ax.axis('off')
    return im, ax