# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 04:54:08 2016

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
from QuantumSystem import QuantumSystem
from plot1D import plot1D
import matplotlib.pyplot as plt
import matplotlib.animation as animate

print('Starting..')
n = [1024]
dx = [1/(ni-1) for ni in n]
b = 'd'
V0 = np.zeros(n)
m = 1
h = 1
print('Intialization Complete..')

S = QuantumSystem(n, dx, b, V0, m, h)
x = S.x[0]
Vh = .5*m*(1/dx[0])**2*(x-x[int(n[0]/2)])**2
S.setPotential(Vh)
print('System Constructed..')
[u, v] = S.energyEigenStates()
print('Eigenvalues and Eigenvectors Calculated..')

k = 512
dt = (2*np.pi*h/u[0])/(k-1)
t = np.linspace(0, (k-1)*dt, k)
print('Temporal Domain Constructed..')

f = v[:, 0]+v[:, 1]
print('Initial Wave Defined..')
F = S.timeEvolution(f, dt, k)
P = S.probDens(F)
print('Time Evolution Complete..')
mX, mP = S.phaseSpace(F)
mT, mV = S.virial(F)
mE = mT+mV
print('Mean Values Calculated..')

plot1D(np.linspace(0, len(u), len(u)), u, 'Energy Levels')
plot1D(x, f, 'Initial Wavefunction vs Position')
plot1D(mX, mP, 'Phase Space')
plot1D(mT, mV, 'Potential vs. Kinetic Energy')
plot1D(t, mE, 'Energy vs Time')
(n, m) = np.shape(P)
fig, ax = plt.subplots()
line, = ax.plot(x, P[:,0])

def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

def frame(i):
    line.set_ydata(P[:, i])
    return line,

animation = animate.FuncAnimation(fig, frame, np.arange(m),
                                  init_func = init, interval = 25,
                                  blit = True)
plt.show()
print('Plots Constructed..')
