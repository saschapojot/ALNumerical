import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slin
from datetime import datetime
#consts
omegaF=10
k0=0.3
k1=1
Gamma=0
phi11s=np.log(1/(
    4*(np.sinh(1/2*(k1+np.conj(k1))))**2
))

tTot=10
Q=1000
dt=tTot/Q

nStart=-256
L=512#lattice length
nRange=range(nStart,nStart+L)
N=int(L/2)

kValsAll=np.fft.fftfreq(N)*2*np.pi#not shifted
def t1(t):
    return 1#np.exp(-1j*omegaF*t)

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
    return (1j*np.exp(-1j*k0-k1)+1j*np.exp(1j*k0+k1))*t
    # return -1/omegaF*np.exp(-1j*k0-k1)*(np.exp(-1j*omegaF*t)-1)\
    #        +1/omegaF*np.exp(1j*k0+k1)*(np.exp(1j*omegaF*t)-1)



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


def h(k,t):
    """

    :param k:
    :param t:
    :return:
    """
    rst=np.zeros((2,2),dtype=complex)

    rst[0,0]=-Gamma-f(t)
    rst[0,1]=-A1(t)*np.exp(1j*k)-A2(t)
    rst[1,0]=-B2(t)*np.exp(-1j*k)-B1(t)
    rst[1,1]=-Gamma-g(t)

    return rst


initVec=[y(n,0) for n in nRange]



def oneStepRk2(wVec,zVec,t):
    '''

    :param wVec:
    :param zVec:
    :return:aq+1,bq+1
    '''

    #initialize k1 and k2 vector
    k1Vec=[0]*L
    k2Vec=[0]*L
    #fill in k1
    for n in range(0,N):
        k1Vec[2*n]=dt*(1j*A1(t)*zVec[(n-1)%N]+1j*A2(t)*zVec[n%N])*np.abs(wVec[n%N])**2
        k1Vec[2*n+1]=dt*(1j*B1(t)*wVec[n%N]+1j*B2(t)*wVec[(n+1)%N])*np.abs(zVec[n%N])**2
    #fill in k2
    for n in range(0,N):
        k2Vec[2*n]=dt*(
                1j*A1(t+dt)*(zVec[(n-1)%N]+k1Vec[(2*n-1)%L])+1j*A2(t+dt)*(zVec[n%N]+k1Vec[(2*n+1)%L])
        )*np.abs(wVec[n%N]+k1Vec[(2*n)%L])**2

        k2Vec[2*n+1]=dt*(
            1j*B1(t+dt)*(wVec[n%N]+k1Vec[2*n%L])+1j*B2(t+dt)*(wVec[(n+1)%N]+k1Vec[(2*n+2)%L])
        )*np.abs(zVec[n%N]+k1Vec[(2*n+1)%L])**2


    #initialize aq+1, bq+1
    aVecNext=[0]*N
    bVecNext=[0]*N
    for n in range(0,N):
        aVecNext[n]=wVec[n]+1/2*(k1Vec[2*n]+k2Vec[2*n])
        bVecNext[n]=zVec[n]+1/2*(k1Vec[2*n+1]+k2Vec[2*n+1])

    return aVecNext,bVecNext



def oneTimeStepEvolution(aVec,bVec,t):
    """

    :param aVec:
    :param bVec:
    :param t:
    :return: aVecNext, bVecNext
    """
    #Step 1
    xVec=np.fft.fft(aVec)
    yVec=np.fft.fft(bVec)
    #Step 2
    #initialize u, v
    uVec=[0]*N
    vVec=[0]*N
    for j in range(0,N):
        kTmp=kValsAll[j]
        hMat=h(kTmp,t)
        vecTmp=np.array([xVec[j],yVec[j]])
        ujTmp,vjTmp=slin.expm(-1j*dt*hMat).dot(vecTmp)
        uVec[j]=ujTmp
        vVec[j]=vjTmp
    #Step 3
    wVec=np.fft.ifft(uVec)
    zVec=np.fft.ifft(vVec)
    #Step 4
    aVecNext,bVecNext=oneStepRk2(wVec,zVec,t)
    return aVecNext,bVecNext


