import numpy as np
import matplotlib.pyplot as plt

#this script plots 2-soliton solution at t=0

k0 = 0.8
a1=0.1
coefPi=0.99
b1=np.pi*coefPi-k0
k1=a1+1j*b1
a2=0.3
b2=0.2
k2=a2+1j*b2

phi11s = np.log(1 / (
        4 * (np.sinh(1 / 2 * (k1 + np.conj(k1)))) ** 2
))
phi12s = np.log(1 / (
        4 * (np.sinh(1 / 2 * (k1 + np.conj(k2)))) ** 2
))

phi21s = np.log(1 / (
        4 * (np.sinh(1 / 2 * (k2 + np.conj(k1)))) ** 2
))

phi22s = np.log(1 / (
        4 * (np.sinh(1 / 2 * (k2 + np.conj(k2)))) ** 2
))

phi12=np.log(4*(np.sinh(1/2*(k1-k2)))**2)

phi1s2s=np.conj(phi12)

phi121s=phi12+phi11s+phi21s
phi122s=phi12+phi12s+phi22s

phi121s2s=phi11s+phi12s+phi21s+phi22s\
    +phi12+phi1s2s

#initial values of m1 and m2
m1=0
m2=-5

def gn1(n):
    return np.exp(k1*n+m1)+np.exp(k2*n+m2)

def fn2(n):
    return np.exp((k1+np.conj(k1))*n+m1+np.conj(m1)+phi11s)\
        +np.exp((k1+np.conj(k2))*n+m1+np.conj(m2)+phi12s)\
        +np.exp((k2+np.conj(k1))*n+m2+np.conj(m1)+phi21s)\
        +np.exp((k2+np.conj(k2))*n+m2+np.conj(m2)+phi22s)


def gn3(n):
    return np.exp((k1+k2+np.conj(k1))*n+m1+m2+np.conj(m1)+phi121s)\
        +np.exp((k1+k2+np.conj(k2))*n+m1+m2+np.conj(m2)+phi122s)



def fn4(n):
    return np.exp((k1+k2+np.conj(k1)+np.conj(k2))*n+m1+m2+np.conj(m1)+np.conj(m2)+phi121s2s)



def psin(n):
    return (gn1(n)+gn3(n))/(1+fn2(n)+fn4(n))*np.exp(1j*k0*n)



halfLength=250
nStart = -150
L = 2 * halfLength  # lattice length
sites = np.array(range(nStart, nStart + L))

ftSize=17
fig=plt.figure()
ax=fig.add_subplot()
truncation=300
ax.plot(sites[:truncation],np.abs(psin(sites))[:truncation],color="black")

ax.set_xlabel("sites",fontsize=ftSize)
ax.set_ylabel("$|\psi_{n}|$",fontsize=ftSize)
ax.tick_params(axis='both', which='major', labelsize=ftSize-3)
ax.set_title("Shape of 2-soliton at t = 0",fontsize=ftSize)

plt.savefig("2Soliton.pdf")
plt.close()