# -*- coding: utf-8 -*-
"""
Created on Sat Apr 02 15:06:01 2016

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
import scipy.integrate as spint


def integrate(dx, f, k):
    """ given sample separation dx (array-like), function f (array-like), and
        dimensions to be integrated over k (array-like), integrates function f
        over the dimensions in k """
    # array of dimensions to integrate over
    k = np.array(k)
    # alias of integrand function
    g = f
    # for loop to step through each integration axis
    for i in xrange(0, len(k)):
        #integrate over dimension
        g = spint.simps(g, None, dx[i], k[i])
        # reduce the dimensions by 1
        k = k - 1
    # return the integral
    return g