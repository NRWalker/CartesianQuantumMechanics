# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 08:03:07 2016

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np


class DiscreteCartesianDomain:
    """ a class used to establish a basic discrete Cartesian coordinate
        system. three paramters are required for initialization. the
        sampleNumber is a list or numpy array that established the number of
        equally spaced discrete samples in each direction. the
        sampleSeparation is a list or array of the same length as the length of
        sampleNumber that establishes exactly what the equal spacing is between
        sample points in each dimension. the boundaryCondition is a string of
        the same length as the length of the sampleNumber and establishes
        the edge behavior of the coordinates. the two conditions supported
        are 'd' for dirichlet boundary conditions and 'p' for periodic boundary
        conditions. mixed conditions in each dimension is supported.

        initialization of a 1D system with 1000 sample points, a spacing of 1,
        and periodic boundary conditions:

        DiscreteCartesianDomain([1000], [1], 'p')

        initialization of a 2D system with 50 sample points in the x direction,
        100 sample points in the y direction, a spacing of 1 in the x
        direction, a spacing of .5 in the y direction, dirichlet conditions in
        the x direction, and periodic conditions in the y direction:

        DiscreteCartesianDomain([50, 100], [1, .5], 'dp') """

    def __init__(self, sampleNumber, sampleSeparation, boundaryCondition):
        """ takes three parameters. sampleNumber (array-like), sampleSeparation
            (array-like), and boundaryCondition (string) """
        # number of sample points
        self.n = np.array(sampleNumber)
        # spacing between sample points
        self.dx = np.array(sampleSeparation)
        # boundary conditions
        self.bc = boundaryCondition
        # construct coordinate system
        self.constructDomain()

    def returnSampleNumber(self):
        """ when called, returns the sample number of the coordinate system
            (array) """
        # return sample number
        return self.n

    def returnSampleSeparation(self):
        """ when called, returns the sample separation of the coordinate
            system (array) """
        # return sample separation
        return self.dx

    def returnBoundaryCondition(self):
        """ when called, returns the boundary conditions of the coordinate
            coordinate system (string) """
        # return boundary condition
        return self.bc

    def returnDimensionLength(self):
        """ when called, returns the length of each direction in the coordinate
            system (array) """
        # return length
        return self.l

    def returnDomain(self):
        """ when called, returns the domain of the coordinate system (list of
            arrays) """
        # return domain
        return self.x

    def constructDomain(self):
        """ when called, constructs the domain of the system using the
            information from the number of sample points, the sample point
            separation, and the boundary conditions """
        # each dimension has n-1 links of length dx, l = dx*(n-1)
        self.l = self.dx*(self.n-1)
        # empty list to store arrays of allowed discrete positions
        self.x = []
        # for each dimension in n
        for i in xrange(0, len(self.n)):
            # periodic boundary conditions will add an extra link
            if self.bc[i] == 'p':
                self.l[i] += self.dx[i]
            # construct dimension-specific array of allowed discrete positions
            d = np.array(np.linspace(0, (self.n[i]-1)*self.dx[i], self.n[i]))
            # append array of allowed positions
            self.x.append(d)

    def setSampleNumber(self, sampleNumber):
        """ when called, takes in the parameter sampleNumber (array-like) and
            uses it to construct a new domain accordingly """
        # reassign sample number
        self.n = np.array(sampleNumber)
        # construct new domain
        self.constructDomain()

    def setSampleSeparation(self, sampleSeparation):
        """ when called, takes in the parameter sampleSeparation (array-like)
            and uses it to construct a new domain accordingly """
        # reassign sample separation
        self.dx = np.array(sampleSeparation)
        # construct new domain
        self.constructDomain()

    def setBoundaryCondition(self, boundaryCondition):
        """ when called, takes in the parameter boundaryCondition (string) and
            uses it to construct a new domain accordingly """
        # reassign boundary condition
        self.bc = boundaryCondition
        # construct new domain
        self.constructDomain()

    def expand(self, sampleNumber, sampleSeparation, boundaryCondition):
        """ when called, takes in the parameters sampleNumber (array-like),
            sampleSepration (array-like), and boundarycondition (string) in
            order to extend the dimension of the coordinate system and
            construct a new domain accordingly """
        # extend the sample number to include the new desired dimension
        self.n = np.concatenate((self.n, np.array(sampleNumber)), 1)
        # extend the sample separation to include the new desired dimension
        self.dx = np.concatenate((self.dx, np.array(sampleSeparation)), 1)
        # extend the boundary conditions to include the new desired dimension
        self.bc = ''.join([self.bc, boundaryCondition])
        # construct the new domain
        self.constructDomain()
