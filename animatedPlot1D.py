# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 00:52:47 2016

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate


def animatedPlot1D(x, F, t = ''):
    (n, m) = np.shape(F)
    fig, ax = plt.subplots()
    line, = ax.plot(x, F[:,0])

    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    def frame(i):
        line.set_ydata(F[:, i])
        return line,

    animation = animate.FuncAnimation(fig, frame, np.arange(m),
                                      init_func = init, interval = 25,
                                      blit = True)
    plt.show()
