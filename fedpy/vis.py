# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def linecomparisonplot(*lines, **kwargs):
    """
    Comparison plot of a few lines
    """
    
    figsize = kwargs.pop('figsize', (6, 4))
    nlines = len(lines)
    x = kwargs.pop('x', range(len(lines[0])))
    xlabel = kwargs.pop('xlabel', '')
    ylabel = kwargs.pop('ylabel', '')
    axlabelsize = kwargs.pop('axlabelsize', 15)
    linelabels = kwargs.pop('linelabels', ['']*nlines)
    islgd = kwargs.pop('legend', False)
    lgdsize = kwargs.pop('lgdtxtsize', 12)
    stylespecs = kwargs.pop('style', '-')
    
    # Plotting routine
    f, ax = plt.subplots(figsize=figsize)
    for i in range(nlines):
        ax.plot(x, lines[i], stylespecs, label=linelabels[i], **kwargs)
    
    ax.set_xlabel(xlabel, fontsize=axlabelsize)
    ax.set_ylabel(ylabel, fontsize=axlabelsize)
    if islgd == True:
        ax.legend(fontsize=lgdsize)
    
    return f, ax
    
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
