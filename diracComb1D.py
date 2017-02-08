# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 23:59:38 2016

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
from QuantumSystem import QuantumSystem
from plot1D import plot1D
from normalize import normalize

print('Starting..')
n = [4096]
dx = [1/(ni-1) for ni in n]
b = 'd'
V0 = np.zeros(n)
m = 1
h = 1
print('Intialization Complete..')
S = QuantumSystem(n, dx, b, V0, m, h)
x = S.x[0]
Vd = lambda i: 1e5*normalize(dx, n, np.exp(-(x-x[i])**2/(2*(dx[0]/1e20)**2)))
j = np.linspace(0, n[0], 22)
V = V0
for i in xrange(1, len(j)-1, 2):
    V = V+Vd(j[i])
S.setPotential(V)
print('System Constructed..')
[u, v] = S.energyEigenStates()
print('Eigenvalues and Eigenvectors Calculated..')

plot1D(np.arange(len(u)), u/np.abs(np.min(u)), 'Energy Levels')
print('Plot Constructed..')
