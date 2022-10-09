import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import mpmath
from mpmath import mp
import pandas as pd
mp.dps = 100

#this script calculates evolution of wavefunction with changing
#linear hopping coefficients and fixed nonlinear coefficient
k0 = 0.8
a1=0.1
coefPi=0.9
b1=np.pi*coefPi-k0
k1=a1+1j*b1
s=12
r=1/s


phi11s = np.log(1 / (
        4 * (np.sinh(1 / 2 * (k1 + np.conj(k1)))) ** 2
))
tTot=100
Q=100000
dt=tTot/Q
print("dt=" + str(dt))
L1=0
L2=0
halfLength=250
nStart = -150
L = 2 * halfLength  # lattice length
initCenter=0
nRange = range(nStart, nStart + L)

def xi(n):
    if n%2==0:
        return np.sqrt(r)
    else:
        return 1/np.sqrt(r)

def y(n, t):
    """

    :param n:
    :param t:
    :return: exact 1-bright-soliton solution
    """

    return 1/2*xi(n)*mpmath.sech(n*a1+L1+1/2*phi11s+(mpmath.exp(-a1)-mpmath.exp(a1))*t*mpmath.sin(k0+b1))\
           *mpmath.exp(1j*(n*(k0+b1)+L2+t*(mpmath.exp(-a1)+mpmath.exp(a1))*mpmath.cos(k0+b1))-1/2*phi11s)

def exactVec(t):
    """

    :param t:
    :return: exect soliton solution at time t
    """
    retVec=np.zeros(L,dtype=complex)
    for n in range(0,L):
        j=nRange[n]
        retVec[n]=complex(y(j,t))
    return retVec

initVecmp=[y(n,0) for n in nRange]
initVec=[complex(elem) for elem in initVecmp]





def A1(t):
    return r


def A2(t):
    return r


def B1(t):
    return s



def B2(t):
    return s

def Cn(n, t):
    if n % 2 == 0:
        return A1(t)#+randomField0[n]
    else:
        return B1(t)#+randomField1[n]

def Dn(n, t):
    if n % 2 == 0:
        return A2(t)#+randomField2[n]
    else:
        return B2(t)#+randomField3[n]


def Fn(n,t):
    return 1

####calc position expectation
def avgPos(psiVec):
    """

    :param psiVec:
    :return: <x>
    """
    rst = 0
    for j in range(0, len(nRange)):
        rst += nRange[j] * np.abs(psiVec[j]) ** 2

    rst /= (np.linalg.norm(psiVec, 2)) ** 2
    return rst


def avgPos2(psiVec):
    """

    :param psiVec:
    :return: <x^{2}>
    """
    rst = 0
    for j in range(0, len(nRange)):
        rst += (nRange[j]) ** 2 * np.abs(psiVec[j]) ** 2
    rst /= (np.linalg.norm(psiVec, 2)) ** 2
    return rst


def norm2(psiVec):
    """

    :param psiVec:
    :return: 2 norm pf psi
    """
    return np.linalg.norm(psiVec, 2)



def height(psiVec):
    """

    :param psiVec:
    :return: maximum height of psi
    """
    # absTmp=[np.abs(elem) for elem in psiVec]
    return np.max(np.abs(psiVec))



def K0Vec(psiVec,q):
    """

    :param psiVec:
    :param q:
    :return:
    """
    tq=q*dt
    retK0 = np.zeros(L, dtype=complex)
    for n in range(0,L):
        retK0[n]=dt*1j*Cn(n,tq)*psiVec[(n-1)%L]+dt*1j*Dn(n,tq)*psiVec[(n+1)%L]\
            +dt*1j*Fn(n,tq)*(psiVec[(n-1)%L]+psiVec[(n+1)%L])*np.abs(psiVec[n])**2


    return retK0



def K1Vec(psiVec,q,K0):
    """

    :param psiVec:
    :param q:
    :param K0:
    :return:
    """
    tq = q * dt

    retK1 = np.zeros(L, dtype=complex)
    for n in range(0,L):
        retK1[n]=dt*1j*Cn(n,tq+1/2*dt)*(psiVec[(n-1)%L]+1/2*K0[(n-1)%L])+dt*1j*Dn(n,tq+1/2*dt)*(psiVec[(n+1)%L]+1/2*K0[(n+1)%L])\
            +dt*1j*Fn(n,tq+1/2*dt)*(psiVec[(n-1)%L]+1/2*K0[(n-1)%L]+psiVec[(n+1)%L]+1/2*K0[(n+1)%L])*np.abs(psiVec[n]+1/2*K0[n])**2

    return retK1


def K2Vec(psiVec, q, K1):
    """

    :param psiVec:
    :param q:
    :param K1:
    :return:
    """
    tq = q * dt
    retK2 = np.zeros(L, dtype=complex)
    for n in range(0,L):
        retK2[n]=dt*1j*Cn(n,tq+1/2*dt)*(psiVec[(n-1)%L]+1/2*K1[(n-1)%L])+dt*1j*Dn(n,tq+1/2*dt)*(psiVec[(n+1)%L]+1/2*K1[(n+1)%L])\
            +dt*1j*Fn(n,tq+1/2*dt)*(psiVec[(n-1)%L]+1/2*K1[(n-1)%L]+psiVec[(n+1)%L]+1/2*K1[(n+1)%L])*np.abs(psiVec[n]+1/2*K1[n])**2

    return retK2


def K3Vec(psiVec, q, K2):
    """

    :param psiVec:
    :param q:
    :param K2:
    :return:
    """
    tq = dt * q
    retK3 = np.zeros(L, dtype=complex)
    for n in range(0,L):
        retK3[n]=dt*1j*Cn(n,tq+dt)*(psiVec[(n-1)%L]+K2[(n-1)%L])+dt*1j*Dn(n,tq+dt)*(psiVec[(n+1)%L]+K2[(n+1)%L])\
            +dt*1j*Fn(n,tq+dt)*(psiVec[(n-1)%L]+K2[(n-1)%L]+psiVec[(n+1)%L]+K2[(n+1)%L])*np.abs(psiVec[n]+K2[n])**2
    return retK3


def oneStepRK4(psiVec, q):
    """

    :param psiVec:
    :param q:
    :return:
    """
    K0 = K0Vec(psiVec, q)
    K1 = K1Vec(psiVec, q, K0)
    K2 = K2Vec(psiVec, q, K1)
    K3 = K3Vec(psiVec, q, K2)
    psiNext = []
    for n in range(0, L):
        psiNext.append(psiVec[n] + 1 / 6 * (K0[n] + 2 * K1[n] + 2 * K2[n] + K3[n]))

    return psiNext
