# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 09:30:37 2016

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
import scipy.sparse as spsp
from DiscreteCartesianDomain import DiscreteCartesianDomain
from fornberg import fornberg


class DifferentiableDomain(DiscreteCartesianDomain):
    """ a class that inherits from the DiscreteCartesianDomain and carries with
        it the exact same data members. two methods are added to encapsulate
        the desired differentiable nature of the coordinate system: the
        derivativeOperator and the laplacianOperator.

        initialization of a 1D system with 1000 sample points, a spacing of 1,
        and periodic boundary conditions:

        DifferentiableDomain([1000], [1], 'p')

        initialization of a 2D system with 50 sample points in the x direction,
        100 sample points in the y direction, a spacing of 1 in the x
        direction, a spacing of .5 in the y direction, dirichlet conditions in
        the x direction, and periodic conditions in the y direction:

        DifferentiableDomain([50, 100], [1, .5], 'dp') """

    def __init__(self, sampleNumber, sampleSeparation, boundaryCondition):
        DiscreteCartesianDomain.__init__(self,
                                         sampleNumber,
                                         sampleSeparation,
                                         boundaryCondition)

    def derivativeOperator(self, v = [-1, 0, 1], o = 1, d = 0):
        """ when called, returns a derivative matrix operator using the
            relative sample points in v, the order o, and the direction d.

            a first-order derivative in the x-direction using only
            nearest-neighbors would be expressed as:

            derivativeOperator([-1, 0, 1], 1, 0)

            a second-order derivative in the y direction using the second-
            nearest neighbors would be expressed as:

            derivativeOperator([-2, -1, 0, 1, 2], 2, 1) """
        # number of samples in direction d
        n = self.n[d]
        # find difference weights (w) on indices (v) centered on 0 of order o
        # using fornberg algorithm
        w = fornberg(0, v, o)
        # prepare on-filled array with same cardinality as dimension
        a = np.ones(n)
        # prepare a list to be filled with arrays of the weights
        u = []
        # loop through weights
        for i in xrange(0, len(v)):
            u.append(a*w[o, i])
        # prepare derivative operator as sparse 1D derivative operator
        D = spsp.spdiags(u, v, n, n, format = 'csc')
        # if periodic boundary conditions
        if self.bc[d] == 'p':
            # find node index
            i = v.index(0)
            # for each row, fill in periodic terms
            for j in xrange(0,i):
                D[j, (n-i+j):n] = w[o, 0:(i-j)]
                D[n-j-1, 0:(i-j)] = w[o, (len(v)-i+j):len(v)]
        # if the domain is of only one dimension
        if len(self.n) != 1:
            # prepare empty list to store each dimension matrix
            A = []
            # loop through dimensions
            for i in xrange(0, len(self.n)):
                # if dimension is being differentiated, use derivative operator
                if i == d:
                    A.append(D)
                # else, simply use identity matrix to preserve the dimension
                # structure
                else:
                    A.append(spsp.eye(self.n[i], self.n[i]))
            # take first kronecker tensor product to compose spaces
            D = spsp.kron(A[1], A[0], format = 'csc')
            # iterate through remaining tensor products to construct full
            # operator
            for i in xrange(2, len(self.n)):
                D = spsp.kron(A[i], D, format = 'csc')
        # return derivative operator, dividing by sample separation to the
        # of the order of the derivative
        return D/self.dx[d-1]**o


    def laplacianOperator(self, v = [-1, 0, 1]):
        """ when called, returns a laplacian matrix operator calculated by
            taking the sum of the second order derivative operators in each
            direction using the relative sample points in v

            a laplacian operator using nearest-neighbors would be expressed as:

            laplacianOperator([-1, 0, 1])

            a laplacian operator using second-nearest-neighbors would be
            expressed as:

            laplacianOperator([-2, -1, 0, 1, 2]) """
        # total number of points
        N = np.prod(self.n)
        # empty sparse matrix to intialize operator
        L = spsp.csc_matrix((N,N))
        # loop through each spatial dimension
        for i in xrange(0, len(self.n)):
            L = L+self.derivativeOperator(v, 2, i)
        return L
