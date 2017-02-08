# -*- coding: utf-8 -*-
"""
Created on Fri May 22 06:59:18 2015

@author: Sifodeas
"""

from __future__ import division, print_function
import numpy as np
import numpy.linalg as nplin
import scipy.integrate as spint
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplin
import copy
import matplotlib.pyplot as plt

def integrate(dx, f, k):

    k = np.array(k) #array of dimensions to integrate over
    g = f #alias of integrand function
    for i in xrange(0, len(k)): #for loop to step through each integration axis
        g = spint.simps(g, None, dx[i], k[i]) #integrate over dimension
        k = k - 1 #reduce the dimensions by 1
    return g #return the integral


def normalize(dx, n, f):

    m = copy.copy(n) #make copy of the dimension of the system
    m.append(-1) #append -1 so that the dimension of g is inferred in the last dimension
    g = np.reshape(np.matrix(np.multiply(np.conj(f), f)), m) #construct PDF from wavefunction f
    g = integrate(dx, g, np.arange(0, len(n), 1))
    if type(g) == np.ndarray:
        f = f / np.sqrt(g[0])
    else:
        f = f / np.sqrt(g)
    return f

def constructDomain(k, m, l, n):

    dt = k / (np.array(m) - 1.)
    t = np.matrix(np.arange(0, k + dt, dt))
    dx = l / (np.array(n) - 1.)
    x = []
    for i in xrange(0, len(n)):
        x.append(np.matrix(np.arange(0, l[i] + dx[i], dx[i])))
    tx = []
    tx.append([dt, t])
    tx.append([dx, x])
    return tx

def derivativeOperator(dx, n, b):

    #2,3D not working, only first dimension correct
    d = [[], []]
    for i in xrange(0, len(n)):
        a = np.ones(n[i])
        d[0].append(spsp.spdiags([-a, a], [-1, 1], n[i], n[i]))
        d[1].append(spsp.spdiags([a, -2 * a, a], [-1, 0, 1], n[i], n[i]))
    for i in xrange(0, len(n)):
        if b[i] == 'P':
            d[0][i] = d[0][i].todense()
            d[0][i][0, n[i] - 1] = -1
            d[0][i][n[i] - 1, 0] = 1
            d[0][i] = spsp.csr_matrix(d[0][i])
            d[1][i] = d[1][i].todense()
            d[1][i][0, n[i] - 1] = 1
            d[1][i][n[i] - 1, 0] = 1
            d[1][i] = spsp.csr_matrix(d[1][i])
    if len(n) == 1:
        D = [[], []]
        D[0].append(d[0][0] / (2 * dx[0]))
        D[1].append(d[1][0] / (np.power(dx[0], 2)))
    else:
        D = [[], []]
        I = [[], []]
        for i in xrange(0, len(n)):
            I[0].append(spsp.eye(n[i], n[i]))
            I[1].append(spsp.eye(n[i], n[i]))
        a = []
        a.append(d)
        for i in xrange(1, len(n)):
            a.append(I)
        for i in xrange(0, len(n)):
            b = spsp.kron(a[1][0][1], a[0][0][0])
            c = spsp.kron(a[1][1][1], a[0][1][0])
            for j in xrange(2, len(n)):
                b = spsp.kron(a[j][0][j], b)
                c = spsp.kron(a[j][1][j], c)
            D[0].append(spsp.csr_matrix(b / (2 * dx[i])))
            D[1].append(spsp.csr_matrix(c / np.power(dx[i],2)))
            a = np.roll(a, 1)
    return D

def laplaceOperator(dx, n, b):

    D = derivativeOperator(dx, n, b)
    L = []
    L.append(D[1][0].todense())
    for i in xrange(1, len(n)):
        L[0] += D[1][i].todense()
    L[0] = spsp.csr_matrix(L[0])
    return L

def positionOperator(dx, n):

    X = [[], []]
    if len(n) == 1:
        a = np.arange(0, (n[0] - 1) * dx[0], dx[0])
        b = np.multiply(a, a)
        X[0].append(spsp.spdiags(a, 0, n[0], n[0]))
        X[1].append(spsp.spdiags(b, 0, n[0], n[0]))
    else:
        x = []
        I = []
        for i in xrange(0, len(n)):
            a = np.arange(0, (n[i] - 1) * dx[i], dx[i])
            x.append(spsp.spdiags(a, 0, n[i], n[i]))
            I.append(spsp.eye(n[i], n[i]))
        a = []
        a.append(x)
        for i in xrange(1, len(n)):
            a.append(I)
        for i in xrange(0, len(n)):
            b = spsp.kron(a[1][1], a[0][0])
            for j in xrange(2, len(n)):
                b = spsp.kron(a[j][j], b)
            X[0].append(spsp.csr_matrix(b))
            X[1].append(spsp.csr_matrix(np.multiply(b, b)))
            a = np.roll(a, 1)
    return X

def momentumOperator(dx, n, b, c):

    h = c[0]
    D = derivativeOperator(dx, n, b)
    P = [[], []]
    for i in xrange(0, len(n)):
        P[0].append(spsp.csr_matrix(-1j * h * D[0][i]))
        P[1].append(spsp.csr_matrix(- np.power(h, 2) * D[1][i]))
    return P

def hamiltonianOperator(dx, n, b, c, v):

    h = c[0]
    m = c[1]
    L = laplaceOperator(dx, n, b)
    V = spsp.spdiags(np.reshape(v, np.prod(n)), 0, np.prod(n), np.prod(n))
    H = []
    H.append(spsp.csc_matrix(-np.power(h, 2) / (2 * m) * L[0] + V))
    return H

def propagationOperator(dx, n, b, c, v, dt):

    h = c[0]
    H = hamiltonianOperator(dx, n, b, c, v)
    H = H[0].todense() / (1j * h)
    UF = spsp.eye(np.prod(n), np.prod(n)).todense() + (dt / 2) * H
    U = []
    U.append(spsp.csr_matrix(nplin.solve(UF, np.conj(UF))))
    return U

def eigenVectors(dx, n, H):

    (u, v) = spsplin.eigsh(H[0], int(np.sqrt(np.prod(n))), None, None, 'SM')
    m = np.shape(v)
    v = np.matrix(v)
    for i in xrange(0, m[1]):
        v[:, i] = normalize(dx, n, v[:, i])
    ev = []
    ev.append(u)
    ev.append(v)
    return ev

def timeEvolution(U, m, f0):

    f0 = np.reshape(f0, [np.prod(np.shape(f0)), 1])
    f = np.matrix(np.zeros([np.prod(np.shape(f0)), m], dtype = complex))
    f[:, 0] = f0
    for i in xrange(1, m):
        f[:, i] = U[0] * f[:, i - 1]
    return f


def expectedValue(dx, x, n, A, f):

    m = np.shape(f)
    nn = copy.copy(n).append(-1)
    e = [np.zeros(m[1], dtype = complex) for i in xrange(len(A))]
    for i in xrange(0, len(A)):
        if type(A[i]) == list:
            for j in xrange(0, len(A[i])):
                for k in xrange(0, m[1]):
                    de = np.multiply(np.conj(f[:, k]), A[i][j] * f[:, k])
                    g = np.reshape(de, nn)
                    g = integrate(dx, g, np.arange(0, len(n), 1))
                    if type(g) == np.ndarray:
                        e[i][k] = g[0]
                    else:
                        e[i][k] = g
        else:
            for k in xrange(0, m[1]):
                    de = np.multiply(np.conj(f[:, k]), A[i] * f[:, k])
                    g = np.reshape(de, nn)
                    g = integrate(dx, g, np.arange(0, len(n), 1))
                    if type(g) == np.ndarray:
                        e[i][k] = g[0]
                    else:
                        e[i][k] = g
    return e


c = [1., 1.] #physical constants hbar and mass
k = 1. #total time
l = [99] #length
m = 100 #time domain resolution
n = [100] #spatial domain resolution
b = 'DD' #boundary conditions

tx = constructDomain(k, m, l, n) #construct spacetime domain
[dt, t] = tx[:][0]
[dx, x] = tx[:][1]
v = np.zeros(np.prod(n))

X = positionOperator(dx, n)
P = momentumOperator(dx, n, b, c)
H = hamiltonianOperator(dx, n, b, c, v)

ev = eigenVectors(dx, n, H)
u = ev[0]
uv = ev[1]

U = propagationOperator(dx, n, b, c, v, dt)

f0 = uv[:, 0]
f0 = normalize(dx, n, f0)
f = timeEvolution(U, m, f0)
p = np.real(np.multiply(np.conj(f), f))

mx = expectedValue(dx, x, n, X, f)
mp = expectedValue(dx, x, n, P, f)
me = expectedValue(dx, x, n, H, f)
sdx = np.sqrt(mx[1] - np.multiply(mx[0], mx[0]))
sdp = np.sqrt(mp[1] - np.multiply(mp[0], mp[0]))
uxp = np.multiply(sdx, sdp)
ee = np.abs(np.real((me[0] - me[0][0]) / me[0][0]))
mee = np.mean(ee)