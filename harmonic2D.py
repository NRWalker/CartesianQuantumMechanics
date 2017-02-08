# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 02:53:29 2016

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
from QuantumSystem import QuantumSystem
from plot1D import plot1D
from wireFramePlot2D import wireFramePlot2D
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print('Starting..')
n = [51, 51]
dx = [1/ni for ni in n]
b = 'dd'
V0 = np.zeros(n)
m = 1
h = 1
print('Initialization Complete..')

S = QuantumSystem(n, dx, b, V0, m, h)
[x, y] = S.x
[X, Y] = np.meshgrid(x, y)
Vh = .5*m*((1/dx[0])**2*(X-x[int(n[0]/2)])**2+
           (1/dx[1])**2*(Y-y[int(n[1]/2)])**2)
S.setPotential(Vh)
print('System Constructed..')
[u, v] = S.energyEigenStates()
print('Eigenvalues and Eigenvectors Calculated..')

k = 512
dt = 10*(2*np.pi*h/u[0])/(k-1)
t = np.linspace(0, (k-1)*dt, k)
print('Temporal Domain Constructed..')

f = v[:, 0]+v[:, 1]
print('Initial Wave Defined..')
F = S.timeEvolution(f, dt, k)
P = S.probDens(F)
print('Time Evolution Complete..')
mX, mPX = S.phaseSpace(F, d = 0)
mY, mPY = S.phaseSpace(F, d = 1)
mT, mV = S.virial(F)
mE = mT+mV
print('Mean Values Calculated..')

plot1D(np.linspace(0, len(u), len(u)), u, 'Energy Levels')
wireFramePlot2D(x, y, f, 'x', 'y', '$\Psi_{00}+\Psi_{01}$')
plot1D(mX, mPX, 'Phase Space x Dimension')
plot1D(mY, mPY, 'Phase Space y Dimension')
plot1D(mT, mV, 'Potential vs. Kinetic Energy')
plot1D(t, mE, 'Energy vs Time')

fig = plt.figure()
ax = axes3d.Axes3D(fig)
Pp = [np.reshape(P[:, i], n) for i in xrange(k)]
mz = np.max(P[:, 0])
surf = ax.plot_wireframe(X, Y, Pp[0])
ax.set_zlim(-1, 1.1*mz)

def update(i, ax, fig):
    ax.cla()
    wframe = ax.plot_wireframe(X, Y, Pp[i])
    ax.set_zlim(0, 1.1*mz)
    return wframe,

ani = animation.FuncAnimation(fig, update, frames = xrange(100),
                              fargs=(ax, fig), interval=100)

plt.show()
print('Plots Constructed')