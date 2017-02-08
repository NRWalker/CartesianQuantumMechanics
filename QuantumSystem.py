# -*- coding: utf-8 -*-
"""
Created on Sat Apr 02 17:08:17 2016

@author: Nicholas
"""

from __future__ import division, print_function
from DifferentiableDomain import DifferentiableDomain
from eigenVectors import eigenVectors
from normalize import normalize
from integrate import integrate
import numpy as np
import scipy.sparse.linalg as spsplin
import scipy.sparse as spsp


class QuantumSystem(DifferentiableDomain):
    """ a class that inherits form the DifferentiableDomain and shares its data
        members. additional data members are the potential (array-like), the
        value of the mass of the particle (float), and the value of hbar
        (float)

        initialization of a 1D system with 1000 sample points, a spacing of 1,
        periodic boundary conditions, a null potential, a mass of 1, and hbar
        set to 1:

        QuantumSystem([1000], [1], 'p', np.zeros(1000), 1, 1)

        initialization of a 2D system with 50 sample points in the x direction,
        100 sample points in the y direction, a spacing of 1 in the x
        direction, a spacing of .5 in the y direction, dirichlet conditions in
        the x direction, periodic conditions in the y direction, a null
        potential, an electron mass, and hbar set to its SI value:

        QuantumSystem([50, 100], [1, .5], 'dp',
                      np.zeros([50,100]), 9.10938356e-31, 1.0545718e-34) """

    def __init__(self, sampleNumber, sampleSeparation,
                 boundaryCondition, potential, mass, hbar):
        """ takes six parameters. sampleNumber (array-like), sampleSeparation
            (array-like), boundaryCondition (string), potential (array-like),
            mass (float), and hbar (float) """
        DifferentiableDomain.__init__(self,
                                         sampleNumber,
                                         sampleSeparation,
                                         boundaryCondition)
        self.V = potential
        self.m = mass
        self.h = hbar

    def returnPotential(self):
        """ when called, returns the system potential (array-like) """
        return self.V

    def returnMass(self):
        """ when called, returns the particle mass (float) """
        return self.m

    def returnHbar(self):
        """ when called, returns the value of hbar (float) """
        return self.hbar

    def setPotential(self, V):
        """ when called, takes in the potential parameter (array-like) and
            updates the system potential accordingly """
        self.V = V

    def setMass(self, m):
        """ when called, takes in the parameter mass (float) and updates the
            particle mass accordingly """
        self.m = m

    def setHbar(self, hbar):
        """ when called, takes in the parameter hbar (float) and updates the
            system value accordingly """
        self.hbar = hbar

    def positionOperator(self, o = 1, d = 0):
        """ when called, returns the position operator of order (power) o (int)
            and a direction d (int)

            the position operator in the x direction would be expressed as:

            positionOperator(1, 0)

            the square position operator in the y direction would be expressed
            as:

            positionOperator(2, 1) """
        X = spsp.spdiags(self.x[d]**o, 0, self.n[d], self.n[d], format = 'csc')
        if len(self.n) != 1:
            # prepare empty list to store each dimension matrix
            A = []
            # loop through dimensions
            for i in xrange(0, len(self.n)):
                # if dimension is being differentiated, use derivative operator
                if i == d:
                    A.append(X)
                # else, simply use identity matrix to preserve the dimension
                # structure
                else:
                    A.append(spsp.eye(self.n[i], self.n[i], format = 'csc'))
            # take first kronecker tensor product to compose spaces
            X = spsp.kron(A[1], A[0])
            # iterate through remaining tensor products to construct full
            # operator
            for i in xrange(2, len(self.n)):
                X = spsp.kron(A[i], X)
        return X

    def momentumOperator(self, v = [-1, 0, 1], o = 1, d = 0):
        """ when called, returns the momentum operator using the relative
            vertices in v (array-like), an order o (int), and a direction d
            (int)

            the momentum operator in the x direction using nearest neighbors
            would be expressed as:

            momentumOperator([-1, 0, 1], 1, 0)

            the square momentum operator in the y direction using
            second-nearest neighbors would be expressed as:

            momentumOperator([-2, -1, 0, 1, 2], 2, 1) """
        return (-1j*self.h)**o*self.derivativeOperator(v, o, d)

    def kineticOperator(self, v = [-1, 0, 1]):
        """ when called, returns the kinetic energy operator using the relative
            vertices in v (array-like)

            the kinetic energy operator using nearest neighbors would be
            expressed as:

            kineticOperator([-1, 0, 1])

            the kinetic energy operator using second-nearest neighbors would
            be expressed as:

            kineticOperator([-2, -1, 0, 1, 2]) """
        return -1*self.h**2/(2*self.m)*self.laplacianOperator(v)

    def potentialOperator(self):
        """ when called, returns the potential energy operator """
        # number of sample points in system
        N = np.prod(self.n)
        # potential values are placed along the diagonal
        return spsp.spdiags(np.reshape(self.V, N), 0, N, N, format = 'csc')

    def hamiltonOperator(self, v = [-1, 0, 1]):
        """ when called, returns the Hamiltonian using the relative vertices in
            v (array-like)

            the Hamiltonian using nearest neighbors would be expressed as:

            hamiltonOperator([-1, 0, 1])

            the Hamiltonian using second-nearest neighbors would be expressed
            as:

            hamiltonOperator([-2, -1, 0, 1, 2]) """
        return self.kineticOperator(v)+self.potentialOperator()

    def propagationOperator(self, dt, v = [-1, 0, 1]):
        """ when called, returns the time porpagation operator using the
            timestep dt (float) and the relative vertices v (array-like)

            the propagation operator using a timestep of 1 and nearest
            neighbors would be expressed as:

            propagationOperator(1, [-1, 0, 1])

            the propagation operator using a timestep of .5 and second-nearest
            neighbors would be expressed as:

            propagationOperator(.5, [-2, -1, 0, 1, 2]) """
        # total number of samples in the system
        N = np.prod(self.n)
        # hamiltonian operator
        H = self.hamiltonOperator(v)
        # hamiltonian operator made anti-hermitian and divided by hbar
        AH = H / (1j * self.h)
        # first order porpagation operator
        UF = spsp.eye(N, N, format = 'csc')+(dt/2)*AH
        # Cayley's form to enforce unitarity
        # http://arxiv.org/pdf/physics/0011068.pdf
        U = spsplin.spsolve(UF, UF.conj())
        return U

    def eigenStates(self, A):
        """ when called, takes in the operator A (array-like) and returns its
            eigenvalues (array-like) and corresponding eigenvectors
            (array-like) in a list """
        return eigenVectors(self.n, self.dx, A)

    def energyEigenStates(self, v = [-1, 0, 1]):
        """ when called, uses the relative vertices in v (array-like) to
            calculate the energy eigenvalues (array-like) and eigenvectors
            (array-like) returned in a list """
        return eigenVectors(self.n, self.dx, self.hamiltonOperator(v))

    def timeEvolution(self, f, dt, k, v = [-1, 0, 1]):
        """ when called, uses an initial wavefunction f (array-like), the
            timestep dt (float), the time sampling k (int), and the relative
            vertices in v (array-like) to calculate the full
            time-evolution of the wavefunction (array-like) """
        # calculate the time propagation operator if one is not provided
        U = self.propagationOperator(dt, v)
        # normalize the wavefunction
        f = normalize(self.dx, self.n, f)
        # prepare the full time-evolution array
        F = np.zeros([len(f), k], dtype = 'complex128')
        # initialize time-evolution array with initial wavefunction
        F[:, 0] = f
        # loop through time samples to generate time-evolution
        for i in xrange(1, k):
            F[:, i] = U*F[:, i-1]
        return F

    def probDens(self, F):
        """ when called, uses a wavefunction F (array-like) to calculate the
            probability density (array-like) """
        return np.multiply(np.conj(F), F)

    def meanValue(self, A, F):
        """ when called, uses an operator A (array-like) and a wavefunction F
            (array-like) to calculate the mean value of the observable
            associated with the operator A with respect to the wavefunction
            F """
        return integrate(self.dx, np.multiply(np.conj(F), A*F), [0])

    def meanPosition(self, F, o = 1, d = 0):
        """ when called, uses a wavefunction F (array-like), an order o (int),
            and a direction d(int) to calculate the mean value of the position
            of order o with respect to the wavefunction F """
        X = self.positionOperator(o, d)
        return self.meanValue(X, F)

    def meanMomentum(self, F, v = [-1, 0, 1], o = 1, d = 0):
        """ when called, uses a wavefunction F (array-like), relative vertices
            v (array-like), an order o (int), and a direction d (int) to
            calculate the mean momentum of order o with respect to the
            wavefunction F """
        P = self.momentumOperator(v, o, d)
        return self.meanValue(P, F)

    def meanKinetic(self, F, v = [-1, 0, 1]):
        """ when called, uses a wavefunction F (array-like) and relative
            vertices v (array-like) to calculate the mean kinetic energy with
            respect to the wavefunction F """
        T = self.kineticOperator(v)
        return self.meanValue(T, F)

    def meanPotential(self, F):
        """ when called, uses a wavefunction F (array-like) to calculate the
            mean potential with repsect to the wavefunction F """
        V = self.potentialOperator()
        return self.meanValue(V, F)

    def meanEnergy(self, F, v = [-1, 0, 1]):
        """ when called, uses a wavefunction F (array-like) and relative
            vertices v (array-like) to calculate the mean energy with respect
            to the wavefunction F """
        H = self.hamiltonOperator(v)
        return self.meanValue(H, F)

    def virial(self, F, v = [-1, 0, 1]):
        """ when called, uses a wavefunction F (array-like) and relative
            vertices v (array-like) to calculate the mean kinetic energy and
            the mean potential energy with respect to the wavefunction F """
        T = self.meanKinetic(F, v)
        V = self.meanPotential(F)
        return T, V

    def phaseSpace(self, F, v = [-1, 0, 1], d = 0):
        """ when called, uses a wavefunction F (array-like), relative vertices
            v (array-like), and direction d (int) to calculate the mean
            position and the mean momentum with repsect to the wavefunction F
            along the direction d """
        X = self.meanPosition(F, 1, d)
        P = self.meanMomentum(F, v, 1, d)
        return X, P
