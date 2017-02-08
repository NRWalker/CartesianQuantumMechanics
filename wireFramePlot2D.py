# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 03:24:28 2016

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def wireFramePlot2D(x, y, z, xl = '', yl = '', t = ''):
    """ given domain x and y (array-like) as well as data z (array-like),
        shows a wireframe plot """
    # initialize figure
    fig = plt.figure()
    fig.suptitle(t)
    # initialize axes with 3D projection
    ax = fig.add_subplot(111, projection='3d')
    # generate meshgrid for plotting
    [X, Y] = np.meshgrid(x, y)
    # reshape data array z to be of the correct dimension for plotting
    Z = np.reshape(z, [len(x), len(y)])
    # generate plot
    ax.plot_wireframe(X, Y, Z)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.suptitle(t)

