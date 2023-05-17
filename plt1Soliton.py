import numpy as np
import matplotlib.pyplot as plt

#this script plots 1-soliton solution at t=0

k0 = 0.8
a1=0.1
coefPi=0.99
b1=np.pi*coefPi-k0
k1=a1+1j*b1
phi11s = np.log(1 / (
        4 * (np.sinh(1 / 2 * (k1 + np.conj(k1)))) ** 2
))


halfLength=250
nStart = -150
L = 2 * halfLength  # lattice length
sites = np.array(range(nStart, nStart + L))


def sech(x):
    """

    :param x:
    :return: sech(x)
    """

    return 1/np.cosh(x)

def psin(n):
    """

    :param n: site number
    :return: wavefunction
    """
    return 1/2*sech(1/2*phi11s+a1*n)*np.exp(1j*(coefPi*n)-1/2*phi11s)



ftSize=17
fig=plt.figure()
ax=fig.add_subplot()
truncation=300
ax.plot(sites[:truncation],np.abs(psin(sites))[:truncation],color="black")

ax.set_xlabel("sites",fontsize=ftSize)
ax.set_ylabel("$|\psi_{n}|$",fontsize=ftSize)
ax.tick_params(axis='both', which='major', labelsize=ftSize-3)
ax.set_title("Shape of 1-soliton at t = 0",fontsize=ftSize)

plt.savefig("1Soliton.pdf")
plt.close()