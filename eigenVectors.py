# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 00:28:15 2016

@author: Nicholas
"""

from __future__ import division, print_function
import scipy.sparse.linalg as spsplin
import numpy as np
from normalize import normalize


def eigenVectors(n, dx, A):
    """ given sample numbering n (array-like), sample separation dx
        (array-like), and an operator A (array-like), calculates the
        eigenvalues (array-like) and the eigenvectors (array-like) and returns
        them in a list """
    # calculate total number of samples
    N = np.prod(n)
    # calculate the first sqrt(N) eigenvalues/vectors
    (u, v) = spsplin.eigsh(A, int(np.sqrt(N)), None, None, 'SM')
    # calculate shape of v
    m = np.shape(v)
    # for each eigenvector
    for i in xrange(0, m[1]):
        # normalize each eigenvector
        v[:, i] = normalize(dx, n, v[:, i])
    # return eigenvalues and eigenvectors in a list
    ev = [u, v]
    return ev