# -*- coding: utf-8 -*-
"""
Created on Sat Apr 02 15:10:27 2016

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
import copy
from integrate import integrate

def normalize(dx, n, f):
    """ given sample separation dx (array-like), sample number (array-like),
        and function f (array-like), normalizes the function f """
    # make copy of the dimension of the system
    m = copy.copy(n)
    # append -1 so that the dimension of g is inferred in the last dimension
    np.append(m, -1)
    # calculate PDF and reshape to integrate along distinct dimensions
    g = np.reshape(np.multiply(np.conj(f), f), m)
    # integrate over PDF
    g = integrate(dx, g, np.arange(0, len(n), 1))
    # cases to habdle the possible types of the integration constant
    if type(g) == np.ndarray:
        f = f / np.sqrt(g[0])
    else:
        f = f / np.sqrt(g)
    # return normalized function
    return f