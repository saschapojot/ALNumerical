import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


#consts
omegaF=10
k0=-0.3
k1=-1
Gamma=0
phi11s=np.log(1/(
    4*(np.sinh(1/2*(k1+np.conj(k1))))**2
))

tTot=10
Q=500
dt=tTot/Q

nStart=-200
L=400#lattice length
nRange=range(nStart,nStart+L)
N=int(L/2)

def t1(t):
    return np.exp(-1j*omegaF*t)

def A1(t):
    return t1(t)

def A2(t):
    return np.conj(t1(t))

def B1(t):
    return t1(t)

def B2(t):
    return np.conj(t1(t))

def f(t):
    return 0

def g(t):
    return 0
def m(t):
    # return (1j*np.exp(-1j*k0-k1)+1j*np.exp(1j*k0+k1))*t
    return -1/omegaF*np.exp(-1j*k0-k1)*(np.exp(-1j*omegaF*t)-1)\
           +1/omegaF*np.exp(1j*k0+k1)*(np.exp(1j*omegaF*t)-1)



def y(n,t):
    """

    :param n:
    :param t:
    :return:
    """
    return np.exp(k1*n+m(t))/(1+\
                              np.exp(
                                  (k1+np.conj(k1))*n+m(t)+np.conj(m(t))+phi11s
                              ))*np.exp(1j*k0*n)

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
        dt*1j*A2(tq)*psiVec[1]*(1+np.abs(psiVec[0])**2)+dt*(1j*Gamma+1j*f(tq))*psiVec[1]
    )
    #1
    retM0.append(
        dt*(1j*B1(tq)*psiVec[0]+1j*B2(tq)*psiVec[2])*(1+np.abs(psiVec[1])**2)\
        +dt*(1j*Gamma+1j*g(tq))*psiVec[1]
    )
    for n in range(1,N-1):
        #2n
        retM0.append(
            dt*(1j*A1(tq)*psiVec[2*n-1]+1j*A2(tq)*psiVec[2*n+1])*(1+np.abs(psiVec[2*n])**2)\
            +dt*(1j*Gamma+1j*f(tq))*psiVec[2*n]
        )
        #2n+1
        retM0.append(
            dt*(1j*B1(tq)*psiVec[2*n]+1j*B2(tq)*psiVec[2*n+2])*(1+np.abs(psiVec[2*n+1])**2)\
            +dt*(1j*Gamma+1j*g(tq))*psiVec[2*n+1]
        )
    #2N-2
    retM0.append(
        dt*(1j*A1(tq)*psiVec[2*N-3]+1j*A2(tq)*psiVec[2*N-1])*(1+np.abs(psiVec[2*N-2])**2)\
        +dt*(1j*Gamma+1j*f(tq))*psiVec[2*N-2]
    )
    #2N-1
    retM0.append(
        dt*1j*B1(tq)*psiVec[2*N-2]*(1+np.abs(psiVec[2*N-1])**2)\
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
        dt*1j*A2(tq+1/2*dt)*(psiVec[1]+1/2*M0[1])*(1+np.abs(psiVec[0]+1/2*M0[0])**2)\
        +dt*(1j*Gamma+1j*f(tq+1/2*dt))*(psiVec[1]+1/2*M0[1])
    )
    #1
    retM1.append(
        dt*(1j*B1(tq+1/2*dt)*(psiVec[0]+1/2*M0[0])+1j*B2(tq+1/2*dt)*(psiVec[2]+1/2*M0[2]))\
        *(1+np.abs(psiVec[1]+1/2*M0[1])**2)\
        +dt*(1j*Gamma+1j*g(tq+1/2*dt))*(psiVec[1]+1/2*M0[1])

    )
    for n in range(1,N-1):
        #2n
        retM1.append(
            dt*(1j*A1(tq+1/2*dt)*(psiVec[2*n-1]+1/2*M0[2*n-1])+1j*A2(tq+1/2*dt)*(psiVec[2*n+1]+1/2*M0[2*n+1]))\
            *(1+np.abs(psiVec[2*n]+1/2*M0[2*n])**2)
            +dt*(1j*Gamma+1j*f(tq+1/2*dt))*(psiVec[2*n]+1/2*M0[2*n])
        )
        #2n+1
        retM1.append(
            dt*(1j*B1(tq+1/2*dt)*(psiVec[2*n]+1/2*M0[2*n])+1j*B2(tq+1/2*dt)*(psiVec[2*n+2]+1/2*M0[2*n+2])) \
            * (1 + np.abs(psiVec[2 * n + 1] + 1 / 2 * M0[2 * n + 1]) ** 2)
            +dt*(1j*Gamma+1j*g(tq+1/2*dt))*(psiVec[2*n+1]+1/2*M0[2*n+1])
        )
    #2N-2
    retM1.append(
        dt*(1j*A1(tq+1/2*dt)*(psiVec[2*N-3]+1/2*M0[2*N-3])+1j*A2(tq+1/2*dt)*(psiVec[2*N-1]+1/2*M0[2*N-1]))\
        *(1+np.abs(psiVec[2*N-2]+1/2*M0[2*N-2])**2)
        +dt*(1j*Gamma+1j*f(tq+1/2*dt))*(psiVec[2*N-2]+1/2*M0[2*N-2])
    )
    #2N-1
    retM1.append(
        dt*1j*B1(tq+1/2*dt)*(psiVec[2*N-2]+1/2*M0[2*N-2])\
        *(1+np.abs(psiVec[2*N-1]+1/2*M0[2*N-1])**2)
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
        dt*1j*A2(tq+1/2*dt)*(psiVec[1]+1/2*M1[1])*(1+np.abs(psiVec[0]+1/2*M1[0])**2)\
        +dt*(1j*Gamma+1j*f(tq+1/2*dt))*(psiVec[1]+1/2*M1[1])
    )
    #1
    retM2.append(
        dt*(1j*B1(tq+1/2*dt)*(psiVec[0]+1/2*M1[0])+1j*B2(tq+1/2*dt)*(psiVec[2]+1/2*M1[2]))\
        *(1+np.abs(psiVec[1]+1/2*M1[1])**2)
        +dt*(1j*Gamma+1j*g(tq+1/2*dt))*(psiVec[1]+1/2*M1[1])
    )

    for n in range(1,N-1):
        #2n
        retM2.append(
            dt*(1j*A1(tq+1/2*dt)*(psiVec[2*n-1]+1/2*M1[2*n-1])+1j*A2(tq+1/2*dt)*(psiVec[2*n+1]+1/2*M1[2*n+1]))\
            *(1+np.abs(psiVec[2*n]+1/2*M1[2*n])**2)
            +dt*(1j*Gamma+1j*f(tq+1/2*dt))*(psiVec[2*n]+1/2*M1[2*n])
        )
        #2n+1
        retM2.append(
            dt*(1j*B1(tq+1/2*dt)*(psiVec[2*n]+1/2*M1[2*n])+1j*B2(tq+1/2*dt)*(psiVec[2*n+2]+1/2*M1[2*n+2]))\
            *(1+np.abs(psiVec[2*n+1]+1/2*M1[2*n+1])**2)
            +dt*(1j*Gamma+1j*g(tq+1/2*dt))*(psiVec[2*n+1]+1/2*M1[2*n+1])
        )
    #2N-2
    retM2.append(
        dt*(1j*A1(tq+1/2*dt)*(psiVec[2*N-3]+1/2*M1[2*N-3])+1j*A2(tq+1/2*dt)*(psiVec[2*N-1]+1/2*M1[2*N-1]))\
        *(1+np.abs(psiVec[2*N-2]+1/2*M1[2*N-2])**2)
        +dt*(1j*Gamma+1j*f(tq+1/2*dt))*(psiVec[2*N-2]+1/2*M1[2*N-2])
    )
    #2N-1
    retM2.append(
        dt*1j*B1(tq+1/2*dt)*(psiVec[2*N-2]+1/2*M1[2*N-2])\
        *(1+np.abs(psiVec[2*N-1]+1/2*M1[2*N-1])**2)
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
        dt*1j*A2(tq+dt)*(psiVec[1]+M2[1])*(1+np.abs(psiVec[0]+M2[0])**2)+dt*(1j*Gamma+1j*f(tq+dt))*(psiVec[1]+M2[1])
    )
    #1
    retM3.append(
        dt*(1j*B1(tq+dt)*(psiVec[0]+M2[0])+1j*B2(tq+dt)*(psiVec[2]+M2[2]))\
        *(1+np.abs(psiVec[1]+M2[1])**2)
        +dt*(1j*Gamma+1j*g(tq+dt))*(psiVec[1]+M2[1])
    )
    for n in range(1,N-1):
        #2n
        retM3.append(
            dt*(1j*A1(tq+dt)*(psiVec[2*n-1]+M2[2*n-1])+1j*A2(tq+dt)*(psiVec[2*n+1]+M2[2*n+1]))\
            *(1+np.abs(psiVec[2*n]+M2[2*n])**2)
            +dt*(1j*Gamma+1j*f(tq+dt))*(psiVec[2*n]+M2[2*n])
        )
        #2n+1
        retM3.append(
            dt*(1j*B1(tq+dt)*(psiVec[2*n]+M2[2*n])+1j*B2(tq+dt)*(psiVec[2*n+2]+M2[2*n+2]))\
            *(1+np.abs(psiVec[2*n+1]+M2[2*n+1])**2)
            +dt*(1j*Gamma+1j*g(tq+dt))*(psiVec[2*n+1]+M2[2*n+1])
        )
    #2N-2
    retM3.append(
        dt*(1j*A1(tq+dt)*(psiVec[2*N-3]+M2[2*N-3])+1j*A2(tq+dt)*(psiVec[2*N-1]+M2[2*N-1]))\
        *(1+np.abs(psiVec[2*N-2]+M2[2*N-2])**2)
        +dt*(1j*Gamma+1j*f(tq+dt))*(psiVec[2*N-2]+M2[2*N-2])
    )
    #2N-1
    retM3.append(
        dt*1j*B1(tq+dt)*(psiVec[2*N-2]+M2[2*N-2])*(1+np.abs(psiVec[2*N-1]+M2[2*N-1])**2)
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
