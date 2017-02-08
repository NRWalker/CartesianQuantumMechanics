# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 01:51:52 2016

@author: Nicholas
"""

from __future__ import division, print_function
import matplotlib.pyplot as plt

def plot1D(x, y, t = ''):
    """ given domain x and data y (array-like) and optional title t (string),
        generates plot """
    # intialize figure
    fig = plt.figure()
    # initialize axes
    ax = fig.add_subplot(111)
    # plot and set title
    ax.plot(x, y)
    plt.suptitle(t)