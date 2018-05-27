# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def imcomparisonplot(*imgs, **kwargs):
    """
    Plot a list of images to compare. The images have the same dimensions
    """

    nim = len(imgs)
    figsize = kwargs.pop('figsize', (7,7))

    xticks = kwargs.pop('xticks', [])
    yticks = kwargs.pop('yticks', [])
    ticks = kwargs.pop('ticks', None)
    if ticks is not None:
        xticks = ticks
        yticks = ticks

    ttl = kwargs.pop('titles', ['']*nim)
    fs = kwargs.pop('fontsize', 15)
    vmins = kwargs.pop('vmins', [])
    vmaxs = kwargs.pop('vmaxs', [])
    cmap = kwargs.pop('colormap', 'RdBu_r')

    f, axs = plt.subplots(1, nim, figsize=figsize)

    for i in range(nim):

        # Obtain the color boundary values
        try:
            vmin = vmins or vmins[i]
            vmax = vmaxs or vmaxs[i]
        except:
            vmin = imgs[i].min()
            vmax = imgs[i].max()

        axs[i].imshow(imgs[i], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        axs[i].set_xticks(xticks)
        axs[i].set_yticks(yticks)
        axs[i].set_title(ttl[i], fontsize=fs)

    return f, axs
