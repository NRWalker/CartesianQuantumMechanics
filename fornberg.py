# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:14:00 2016

@author: Nicholas
"""

from __future__ import division, print_function
from numpy import zeros, arange


def fornberg(u, v, k):
    """ given the index u (int), the relative indices v (array-like), and the
        order k (int), calculates an array of coefficients for derivatives
        through order k

        Generation of Finite Difference Formulas on Arbitrarily Spaced Grids
        Bengt Fornberg

        http://amath.colorado.edu/faculty/fornberg/
        Docs/MathComp_88_FD_formulas.pdf """

    n = len(v)
    C = zeros([k+1, n])
    c1 = 1
    c4 = v[0]-u
    C[0,0] = 1
    for i in xrange(2, n+1):
        mn = min(i, k+1)
        c2 = 1
        c5 = c4
        c4 = v[i-1]-u
        for j in xrange(1, i):
            c3 = v[i-1]-v[j-1]
            c2 = c2*c3
            if j == (i-1):
                C[1:mn, i-1] = c1*(arange(1, mn)*C[0:(mn-1), i-2]-
                               c5*C[1:mn, i-2])/c2
                C[0, i-1] = -c1*c5*C[0, i-2]/c2
            C[1:mn, j-1] = (c4*C[1:mn, j-1]-arange(1, mn)*C[0:(mn-1), j-1])/c3
            C[0, j-1] = c4*C[0, j-1]/c3
        c1 = c2
    return C