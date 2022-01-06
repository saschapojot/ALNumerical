import numpy as mpmath
from datetime import datetime
import matplotlib.pyplot as plt
import mpmath
from mpmath import mp
mp.dps=1000

#consts
omegaF=10
k0=-0.3
k1=-1+0.3j
k2=5-2.4j
Gamma=0
phi11s=mpmath.log(1/(
    4*(mpmath.sinh(1/2*(k1+mpmath.conj(k1))))**2
))
phi12s=mpmath.log(1/(
    4*(mpmath.sinh(1/2*(k1+mpmath.conj(k2))))**2
))
phi21s=mpmath.log(1/(
    4*(mpmath.sinh(1/2*(k2+mpmath.conj(k1))))**2
))
phi22s=mpmath.log(1/(
    4*(mpmath.sinh(1/2*(k2+mpmath.conj(k2))))**2
))
phi12=mpmath.log(
    4*(mpmath.sinh(1/2*(k1-k2)))**2
)
phi1s2s=mpmath.conj(phi12)
phi6=phi11s+phi12+phi12s+phi21s+phi22s+phi1s2s
phi121s=phi12+phi11s+phi21s
phi122s=phi12+phi12s+phi22s
tTot=10
Q=500
dt=tTot/Q

nStart=-200
L=400#lattice length
nRange=range(nStart,nStart+L)
N=int(L/2)

def t1(t):
    return mpmath.exp(-1j*omegaF*t)

def A1(t):
    return t1(t)

def A2(t):
    return mpmath.conj(t1(t))

def B1(t):
    return t1(t)

def B2(t):
    return mpmath.conj(t1(t))

def f(t):
    return 0

def g(t):
    return 0
##1-soliton funcs
def m(t):
    # return (1j*mpmath.exp(-1j*k0-k1)+1j*mpmath.exp(1j*k0+k1))*t
    return -1/omegaF*mpmath.exp(-1j*k0-k1)*(mpmath.exp(-1j*omegaF*t)-1)\
           +1/omegaF*mpmath.exp(1j*k0+k1)*(mpmath.exp(1j*omegaF*t)-1)




def y(n,t):
    """

    :param n:
    :param t:
    :return:
    """
    return mpmath.exp(k1*n+m(t))/(1+\
                              mpmath.exp(
                                  (k1+mpmath.conj(k1))*n+m(t)+mpmath.conj(m(t))+phi11s
                              ))*mpmath.exp(1j*k0*n)




###2-soliton funcs
def m1(t):
    return -1/omegaF*mpmath.exp(-1j*k0-k1)*(mpmath.exp(-1j*omegaF*t)-1)\
           +1/omegaF*mpmath.exp(1j*k0+k1)*(mpmath.exp(1j*omegaF*t)-1)
def m2(t):
    return -1/omegaF*mpmath.exp(-1j*k0-k2)*(mpmath.exp(-1j*omegaF*t)-1)\
            +1/omegaF*mpmath.exp(1j*k0+k2)*(mpmath.exp(1j*omegaF*t)-1)

def gn1(n,t):
    return mpmath.exp(k1*n+m1(t))+mpmath.exp(k2*n+m2(t))
def gn3(n,t):
    return mpmath.exp((k1+k2+mpmath.conj(k1))*n+m1(t)+m2(t)+mpmath.conj(m1(t))+phi121s)\
            +mpmath.exp((k1+k2+mpmath.conj(k2))*n+m1(t)+m2(t)+mpmath.conj(m2(t))+phi122s)
def fn2(n,t):
    return mpmath.exp((k1+mpmath.conj(k1))*n+m1(t)+mpmath.conj(m1(t))+phi11s)\
            +mpmath.exp((k1+mpmath.conj(k2))*n+m1(t)+mpmath.conj(m2(t))+phi12s)\
            +mpmath.exp((k2+mpmath.conj(k1))*n+m2(t)+mpmath.conj(m1(t))+phi21s)\
            +mpmath.exp((k2+mpmath.conj(k2))*n+m2(t)+mpmath.conj(m2(t))+phi22s)

def fn4(n,t):
    return mpmath.exp((k1+k2+mpmath.conj(k1+k2))*n+m1(t)+m2(t)+mpmath.conj(m1(t)+m2(t))+phi6)
def psi2Soliton(n,t):
    return (gn1(n,t)+gn3(n,t))/(1+fn2(n,t)+fn4(n,t))*mpmath.exp(1j*k0*n)
##########################RK4 part
def M0Vec(psiVec,q):
    """

    :param psiVec:
    :param q: time step
    :return:
    """

    retM0=[]
    tq=dt*q
    #0
    retM0.append(
        dt*1j*A2(tq)*psiVec[1]*(1+mpmath.fabs(psiVec[0])**2)+dt*(1j*Gamma+1j*f(tq))*psiVec[1]
    )
    #1
    retM0.append(
        dt*(1j*B1(tq)*psiVec[0]+1j*B2(tq)*psiVec[2])*(1+mpmath.fabs(psiVec[1])**2)\
        +dt*(1j*Gamma+1j*g(tq))*psiVec[1]
    )
    for n in range(1,N-1):
        #2n
        retM0.append(
            dt*(1j*A1(tq)*psiVec[2*n-1]+1j*A2(tq)*psiVec[2*n+1])*(1+mpmath.fabs(psiVec[2*n])**2)\
            +dt*(1j*Gamma+1j*f(tq))*psiVec[2*n]
        )
        #2n+1
        retM0.append(
            dt*(1j*B1(tq)*psiVec[2*n]+1j*B2(tq)*psiVec[2*n+2])*(1+mpmath.fabs(psiVec[2*n+1])**2)\
            +dt*(1j*Gamma+1j*g(tq))*psiVec[2*n+1]
        )
    #2N-2
    retM0.append(
        dt*(1j*A1(tq)*psiVec[2*N-3]+1j*A2(tq)*psiVec[2*N-1])*(1+mpmath.fabs(psiVec[2*N-2])**2)\
        +dt*(1j*Gamma+1j*f(tq))*psiVec[2*N-2]
    )
    #2N-1
    retM0.append(
        dt*1j*B1(tq)*psiVec[2*N-2]*(1+mpmath.fabs(psiVec[2*N-1])**2)\
        +dt*(1j*Gamma+1j*g(tq))*psiVec[2*N-1]
    )
    return retM0


def M1Vec(psiVec,q,M0):
    """

    :param psiVec:
    :param q:
    :param M0:
    :return:
    """
    retM1=[]
    tq=q*dt

    #0
    retM1.append(
        dt*1j*A2(tq+1/2*dt)*(psiVec[1]+1/2*M0[1])*(1+mpmath.fabs(psiVec[0]+1/2*M0[0])**2)\
        +dt*(1j*Gamma+1j*f(tq+1/2*dt))*(psiVec[1]+1/2*M0[1])
    )
    #1
    retM1.append(
        dt*(1j*B1(tq+1/2*dt)*(psiVec[0]+1/2*M0[0])+1j*B2(tq+1/2*dt)*(psiVec[2]+1/2*M0[2]))\
        *(1+mpmath.fabs(psiVec[1]+1/2*M0[1])**2)\
        +dt*(1j*Gamma+1j*g(tq+1/2*dt))*(psiVec[1]+1/2*M0[1])

    )
    for n in range(1,N-1):
        #2n
        retM1.append(
            dt*(1j*A1(tq+1/2*dt)*(psiVec[2*n-1]+1/2*M0[2*n-1])+1j*A2(tq+1/2*dt)*(psiVec[2*n+1]+1/2*M0[2*n+1]))\
            *(1+mpmath.fabs(psiVec[2*n]+1/2*M0[2*n])**2)
            +dt*(1j*Gamma+1j*f(tq+1/2*dt))*(psiVec[2*n]+1/2*M0[2*n])
        )
        #2n+1
        retM1.append(
            dt*(1j*B1(tq+1/2*dt)*(psiVec[2*n]+1/2*M0[2*n])+1j*B2(tq+1/2*dt)*(psiVec[2*n+2]+1/2*M0[2*n+2])) \
            * (1 + mpmath.fabs(psiVec[2 * n + 1] + 1 / 2 * M0[2 * n + 1]) ** 2)
            +dt*(1j*Gamma+1j*g(tq+1/2*dt))*(psiVec[2*n+1]+1/2*M0[2*n+1])
        )
    #2N-2
    retM1.append(
        dt*(1j*A1(tq+1/2*dt)*(psiVec[2*N-3]+1/2*M0[2*N-3])+1j*A2(tq+1/2*dt)*(psiVec[2*N-1]+1/2*M0[2*N-1]))\
        *(1+mpmath.fabs(psiVec[2*N-2]+1/2*M0[2*N-2])**2)
        +dt*(1j*Gamma+1j*f(tq+1/2*dt))*(psiVec[2*N-2]+1/2*M0[2*N-2])
    )
    #2N-1
    retM1.append(
        dt*1j*B1(tq+1/2*dt)*(psiVec[2*N-2]+1/2*M0[2*N-2])\
        *(1+mpmath.fabs(psiVec[2*N-1]+1/2*M0[2*N-1])**2)
        +dt*(1j*Gamma+1j*g(tq+1/2*dt))*(psiVec[2*N-1]+1/2*M0[2*N-1])
    )

    return retM1


def M2Vec(psiVec,q,M1):
    """

    :param psiVec:
    :param q:
    :param M1:
    :return:
    """
    retM2=[]
    tq=q*dt
    #0
    retM2.append(
        dt*1j*A2(tq+1/2*dt)*(psiVec[1]+1/2*M1[1])*(1+mpmath.fabs(psiVec[0]+1/2*M1[0])**2)\
        +dt*(1j*Gamma+1j*f(tq+1/2*dt))*(psiVec[1]+1/2*M1[1])
    )
    #1
    retM2.append(
        dt*(1j*B1(tq+1/2*dt)*(psiVec[0]+1/2*M1[0])+1j*B2(tq+1/2*dt)*(psiVec[2]+1/2*M1[2]))\
        *(1+mpmath.fabs(psiVec[1]+1/2*M1[1])**2)
        +dt*(1j*Gamma+1j*g(tq+1/2*dt))*(psiVec[1]+1/2*M1[1])
    )

    for n in range(1,N-1):
        #2n
        retM2.append(
            dt*(1j*A1(tq+1/2*dt)*(psiVec[2*n-1]+1/2*M1[2*n-1])+1j*A2(tq+1/2*dt)*(psiVec[2*n+1]+1/2*M1[2*n+1]))\
            *(1+mpmath.fabs(psiVec[2*n]+1/2*M1[2*n])**2)
            +dt*(1j*Gamma+1j*f(tq+1/2*dt))*(psiVec[2*n]+1/2*M1[2*n])
        )
        #2n+1
        retM2.append(
            dt*(1j*B1(tq+1/2*dt)*(psiVec[2*n]+1/2*M1[2*n])+1j*B2(tq+1/2*dt)*(psiVec[2*n+2]+1/2*M1[2*n+2]))\
            *(1+mpmath.fabs(psiVec[2*n+1]+1/2*M1[2*n+1])**2)
            +dt*(1j*Gamma+1j*g(tq+1/2*dt))*(psiVec[2*n+1]+1/2*M1[2*n+1])
        )
    #2N-2
    retM2.append(
        dt*(1j*A1(tq+1/2*dt)*(psiVec[2*N-3]+1/2*M1[2*N-3])+1j*A2(tq+1/2*dt)*(psiVec[2*N-1]+1/2*M1[2*N-1]))\
        *(1+mpmath.fabs(psiVec[2*N-2]+1/2*M1[2*N-2])**2)
        +dt*(1j*Gamma+1j*f(tq+1/2*dt))*(psiVec[2*N-2]+1/2*M1[2*N-2])
    )
    #2N-1
    retM2.append(
        dt*1j*B1(tq+1/2*dt)*(psiVec[2*N-2]+1/2*M1[2*N-2])\
        *(1+mpmath.fabs(psiVec[2*N-1]+1/2*M1[2*N-1])**2)
        +dt*(1j*Gamma+1j*g(tq+1/2*dt))*(psiVec[2*N-1]+1/2*M1[2*N-1])
    )

    return retM2


def M3Vec(psiVec,q,M2):
    """
    :param psiVec:
    :param q:
    :param M2:
    :return:
    """
    tq=q*dt
    retM3=[]
    #0
    retM3.append(
        dt*1j*A2(tq+dt)*(psiVec[1]+M2[1])*(1+mpmath.fabs(psiVec[0]+M2[0])**2)+dt*(1j*Gamma+1j*f(tq+dt))*(psiVec[1]+M2[1])
    )
    #1
    retM3.append(
        dt*(1j*B1(tq+dt)*(psiVec[0]+M2[0])+1j*B2(tq+dt)*(psiVec[2]+M2[2]))\
        *(1+mpmath.fabs(psiVec[1]+M2[1])**2)
        +dt*(1j*Gamma+1j*g(tq+dt))*(psiVec[1]+M2[1])
    )
    for n in range(1,N-1):
        #2n
        retM3.append(
            dt*(1j*A1(tq+dt)*(psiVec[2*n-1]+M2[2*n-1])+1j*A2(tq+dt)*(psiVec[2*n+1]+M2[2*n+1]))\
            *(1+mpmath.fabs(psiVec[2*n]+M2[2*n])**2)
            +dt*(1j*Gamma+1j*f(tq+dt))*(psiVec[2*n]+M2[2*n])
        )
        #2n+1
        retM3.append(
            dt*(1j*B1(tq+dt)*(psiVec[2*n]+M2[2*n])+1j*B2(tq+dt)*(psiVec[2*n+2]+M2[2*n+2]))\
            *(1+mpmath.fabs(psiVec[2*n+1]+M2[2*n+1])**2)
            +dt*(1j*Gamma+1j*g(tq+dt))*(psiVec[2*n+1]+M2[2*n+1])
        )
    #2N-2
    retM3.append(
        dt*(1j*A1(tq+dt)*(psiVec[2*N-3]+M2[2*N-3])+1j*A2(tq+dt)*(psiVec[2*N-1]+M2[2*N-1]))\
        *(1+mpmath.fabs(psiVec[2*N-2]+M2[2*N-2])**2)
        +dt*(1j*Gamma+1j*f(tq+dt))*(psiVec[2*N-2]+M2[2*N-2])
    )
    #2N-1
    retM3.append(
        dt*1j*B1(tq+dt)*(psiVec[2*N-2]+M2[2*N-2])*(1+mpmath.fabs(psiVec[2*N-1]+M2[2*N-1])**2)
        +dt*(1j*Gamma+1j*g(tq+dt))*(psiVec[2*N-1]+M2[2*N-1])
    )
    return retM3


def oneStepRK4(psiVec,q):
    """

    :param psiVecIn:
    :param q:
    :return:
    """
    M0=M0Vec(psiVec,q)
    M1=M1Vec(psiVec,q,M0)
    M2=M2Vec(psiVec,q,M1)
    M3=M3Vec(psiVec,q,M2)
    psiNext=[]
    for j in range(0,L):
        psiNext.append(psiVec[j]+1/6*(M0[j]+2*M1[j]+2*M2[j]+M3[j]))

    return psiNext
