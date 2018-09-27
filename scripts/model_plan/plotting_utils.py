#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']


def turn_off_axes_labels(ax):
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        right='off',         # ticks along the top edge are off
        labelleft='off') # labels along the bottom edge are off


def turn_off_box(ax):
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        top='off'          # ticks along the top edge are off
    )

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        right='off',         # ticks along the top edge are off
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_tick_size(ax, size):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(size)
