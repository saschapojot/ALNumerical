import  numpy as np
import matplotlib.pyplot as plt
from datetime import  datetime
from pathlib import Path
import  mpmath
from mpmath import mp
mp.dps=50
#calculates pumping for real t1 != t2
k0=-0.3
k1=1+0.3j
t1Val=2
t2Val=2.01
s=1/2*(t1Val+t2Val)
Gamma=0

phi11s=mpmath.log(1/(
    4*(mpmath.sinh(1/2*(k1+mpmath.conj(k1))))**2
))
tTot=1
Q=50
dt=tTot/Q
print("dt="+str(dt))
nStart=-10
L=20#lattice length
nRange=range(nStart,nStart+L)
N=int(L/2)


def f(t):
    return 0
def g(t):
    return 0

def A1(t):
    return t1Val
def A2(t):
    return t2Val
def B1(t):
    return t1Val
def B2(t):
    return t2Val
def m(t):
    return 1j*mpmath.exp(-1j*k0-k1)*s*t+1j*mpmath.exp(1j*k0+k1)*s*t



def y(n,t):
    """

    :param n:
    :param t:
    :return: exact 1-bright-soliton solution
    """
    return mpmath.exp(k1*n+m(t))/(1+
                              mpmath.exp((k1+mpmath.conj(k1))*n+m(t)+mpmath.conj(m(t))+phi11s
                                     )
                              )*mpmath.exp(1j*k0*n+1j*Gamma*t)
####calc position expectation
def avgPos(psiVec):
    """

    :param psiVec:
    :return: <x>
    """
    rst=0
    for j in range(0,len(nRange)):
        rst+=nRange[j]*mpmath.fabs(psiVec[j])**2
    rst/=(mpmath.norm(psiVec,2))**2
    return rst
def avgPos2(psiVec):
    """

    :param psiVec:
    :return: <x^{2}>
    """
    rst=0
    for j in range(0,len(nRange)):
        rst+=(nRange[j])**2*mpmath.fabs(psiVec[j])**2
    rst/=(mpmath.norm(psiVec,2))**2
    return rst

def norm2(psiVec):
    """

    :param psiVec:
    :return: 2 norm pf psi
    """
    return mpmath.norm(psiVec,2)
def height(psiVec):
    """

    :param psiVec:
    :return: maximum height of psi
    """
    absTmp=[mpmath.fabs(elem) for elem in psiVec]
    return np.max(np.array(absTmp))
##RK4 part
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
