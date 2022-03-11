import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import mpmath
from mpmath import mp

mp.dps = 100

# this script calculates evolution of wavefunction with site-dependent coefficients
# calculates pumping for real t1 != t2
k0 = 1.8
a1=0.4
b1=np.pi-k0
k1=a1+1j*b1
s = 11

Gmax = 5
Gamma = 0

phi11s = np.log(1 / (
        4 * (np.sinh(1 / 2 * (k1 + np.conj(k1)))) ** 2
))
tQuarter = 100
tTot = 4 * tQuarter
Q = 50000*2
dt = tTot / Q
print("dt=" + str(dt))
nStart = -2000
L = 2 * 2000  # lattice length
nRange = range(nStart, nStart + L)

# np.random.seed(0)

def m(t):
    return  1j * np.exp(-1j * k0 - k1) * s * t + 1j * np.exp(1j * k0 + k1) * s * t

def y(n, t):
    """

    :param n:
    :param t:
    :return: exact 1-bright-soliton solution
    """
    return mpmath.exp(k1 * n + m(t)) / (1 +
                                        mpmath.exp((k1 + np.conj(k1)) * n + m(t) + mpmath.conj(m(t)) + phi11s
                                                   )
                                        ) * mpmath.exp(1j * k0 * n )

initVecmp=[y(n,0) for n in nRange]
initVec=[complex(elem) for elem in initVecmp]
# mu=0
# ratio=10
# sgm=np.max(np.abs(initVec))/ratio
# randomField0=np.random.normal(mu,sgm,L)
# randomField1=np.random.normal(mu,sgm,L)
# randomField2=np.random.normal(mu,sgm,L)
# randomField3=np.random.normal(mu,sgm,L)
# onSiteRandom0=np.random.normal(mu,sgm,L)
# onSiteRandom1=np.random.normal(mu,sgm,L)
def f(t):
    return 0


def g(t):
    return 0


def minusFuncFirstQuarter(t):
    return s - Gmax / tQuarter * t

def plusFuncThirdQuarter(t):
    return s-Gmax+Gmax/tQuarter*(t-2*tQuarter)
########################################################
def plusFunctionFirstQuarter(t):
    return s + Gmax / tQuarter * t

def minusFuncThirdQuarter(t):
    return s+Gmax-Gmax/tQuarter*(t-2*tQuarter)

def A1(t):
    if t < tQuarter:
        return minusFuncFirstQuarter(t)
    elif tQuarter<=t and t<2*tQuarter:
        return s - Gmax
    elif 2*tQuarter<=t and t<3*tQuarter:
        return plusFuncThirdQuarter(t)
    else:#last quarter
        return s


def A2(t):
    if t < tQuarter:
        return plusFunctionFirstQuarter(t)
    elif tQuarter<=t and t<2*tQuarter:
        return s + Gmax
    elif 2*tQuarter<=t and t<3*tQuarter:
        return minusFuncThirdQuarter(t)
    else: #last quarter
        return s


def B1(t):
    return A2(t)

def B2(t):
    return A1(t)





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

strength=0
def impurity(n):
    if n==200:
        return strength
    else:
        return 0


def En(n, t):
    if n % 2 == 0:
        return Gamma + f(t)+impurity(n)#+onSiteRandom0[n]
    else:
        return Gamma + g(t)+impurity(n)#+onSiteRandom1[n]




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


# RK4 part

def K0Vec(psiVec, q):
    """

    :param psiVec:
    :param q:
    :return:
    """
    tq = dt * q
    retK0 = np.zeros(L, dtype=complex)
    for n in range(0, L):
        retK0[n] = dt * 1j * (Cn(n, tq) * psiVec[(n - 1) % L] + Dn(n, tq) * psiVec[(n + 1) % L]) * (
                    1 + np.abs(psiVec[n]) ** 2) \
                   + dt * 1j * En(n, tq) * psiVec[n]

    return retK0


def K1Vec(psiVec, q, K0):
    """

    :param psiVec:
    :param q:
    :param K0:
    :return:
    """
    tq = q * dt

    retK1 = np.zeros(L, dtype=complex)
    for n in range(0, L):
        retK1[n] = dt * 1j * (Cn(n, tq + 1 / 2 * dt) * (psiVec[(n - 1) % L] + 1 / 2 * K0[(n - 1) % L])
                              + Dn(n, tq + 1 / 2 * dt) * (psiVec[(n + 1) % L] + 1 / 2 * K0[(n + 1) % L])
                              ) * (1 + np.abs(psiVec[n] + 1 / 2 * K0[n]) ** 2) \
                   + dt * 1j * En(n, tq + 1 / 2 * dt) * (psiVec[n] + 1 / 2 * K0[n])

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

    for n in range(0, L):
        retK2[n] = dt * 1j * (
                Cn(n, tq + 1 / 2 * dt) * (psiVec[(n - 1) % L] + 1 / 2 * K1[(n - 1) % L]) \
                + Dn(n, tq + 1 / 2 * dt) * (psiVec[(n + 1) % L] + 1 / 2 * K1[(n + 1) % L])
        ) * (1 + np.abs(psiVec[n] + 1 / 2 * K1[n]) ** 2) \
                   + dt * 1j * En(n, tq + 1 / 2 * dt) * (psiVec[n] + 1 / 2 * K1[n])

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
    for n in range(0, L):
        retK3[n] = dt * 1j * (
                Cn(n, tq + dt) * (psiVec[(n - 1) % L] + K2[(n - 1) % L]) \
                + Dn(n, tq + dt) * (psiVec[(n + 1) % L] + K2[(n + 1) % L])
        ) * (1 + np.abs(psiVec[n] + K2[n]) ** 2) \
                   + dt * 1j * En(n, tq + dt) * (psiVec[n] + K2[n])

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
