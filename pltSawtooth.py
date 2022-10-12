import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

#this script plots the evolution of soliton-like solution in 3d plot, and plots the variation of width
inDir="./checkingSawtooth/coef0.9/0s12tTot100a10.1coefPi0.9Q100000/"
inFileName=inDir+"tTot=100data.csv"
tReadCsvStart=datetime.now()
inData=pd.read_csv(inFileName,header=None)
tReadCsvEnd=datetime.now()
print("reading csv: ",tReadCsvEnd-tReadCsvStart)
nRow, L=inData.shape

tTot=100
Q=nRow-1

dt=tTot/Q

fig=plt.figure(figsize=(20,10))

ax1=fig.add_subplot(2,2,1,projection="3d")

sites=list(range(0,L))


#convert str to complex
def strVec2ComplexVec(row):
    retVec=[]
    row=np.array(row)
    for elem in row:
        retVec.append(complex(elem))
    return retVec
def strVec2ComplexVecAbs(row):
    retVec=[]
    row=np.array(row)
    for elem in row:
        retVec.append(np.abs(complex(elem)))
    return retVec

pltQVals=list(range(0,Q,10000))
pltQVals.append(Q)
t3dStart=datetime.now()
truncation=300
for q in pltQVals:
    tValsTmp=[dt*q]*L
    ax1.plot(sites[:truncation],tValsTmp[:truncation],strVec2ComplexVecAbs(inData.iloc[q,])[:truncation],color="black")

pltQRedVals=list([0,Q])
pltQRedVals.append(Q)

for q in pltQRedVals:
    tValsTmp = [dt * q] * L
    ax1.plot(sites[:truncation], tValsTmp[:truncation], strVec2ComplexVecAbs(inData.iloc[q,])[:truncation], color="red")

t3dEnd=datetime.now()
ftSize=17
ax1.view_init(60, -150)
ax1.set_xlim((0,truncation))
ax1.set_xlabel("site",fontsize=ftSize,rotation=60,labelpad=20)
ax1.set_ylabel("time",fontsize=ftSize,rotation=-30,labelpad=10)
ax1.set_zlabel("$|\psi_{n}|$",fontsize=ftSize,labelpad=10)
ax1.set_title("evolution of wavepacket",fontsize=ftSize)
print("3d time: ",t3dEnd-t3dStart)
#calculates sd
def avgPos(psiVec):
    """

    :param psiVec:
    :return: <x>
    """
    rst = 0
    for j in range(0, L):
        rst += j * np.abs(psiVec[j]) ** 2

    rst /= (np.linalg.norm(psiVec, 2)) ** 2
    return rst

def avgPos2(psiVec):
    """

    :param psiVec:
    :return: <x^{2}>
    """
    rst = 0
    for j in range(0, L):
        rst += j ** 2 * np.abs(psiVec[j]) ** 2
    rst /= (np.linalg.norm(psiVec, 2)) ** 2
    return rst
def sd(psiVec):
    x=avgPos(psiVec)
    x2=avgPos2(psiVec)
    sd=np.sqrt(np.abs(x2-x**2))
    return sd

def skewness(psiVec):
    x = avgPos(psiVec)
    x2 = avgPos2(psiVec)
    sd = np.sqrt(np.abs(x2 - x ** 2))
    mu3=0
    for j in range(0,L):
        mu3+=((j-x)/sd)**3*np.abs(psiVec[j])**2
    return mu3

tWidthStart=datetime.now()
pltWidthQVals=list(range(0,Q,200))
pltWidthQVals.append(Q)
widthVals=[sd(strVec2ComplexVec(inData.iloc[q,])) for q in pltWidthQVals]

tWidthVals=[q*dt for q in  pltWidthQVals]
ax2=fig.add_subplot(2,2,2)
ax2.plot(tWidthVals,widthVals,color="black")
ax2.set_xlabel("time",fontsize=ftSize)
ax2.set_ylabel("width",fontsize=ftSize,labelpad=5)
ax2.set_ylim((8.9,9.12))
ax2.set_title("variation of width",fontsize=ftSize)

# ax2.set_yticks(range(8,11))

tWidthEnd=datetime.now()
print("width time: ", tWidthEnd-tWidthStart)

tSkewnessStart=datetime.now()
pltSkewnessQVals=list(range(0,Q,200))
pltSkewnessQVals.append(Q)
tSkewnessVals=[q*dt for q in pltSkewnessQVals]
skewnessVals=[skewness(strVec2ComplexVec(inData.iloc[q,])) for q in pltSkewnessQVals]
ax3=fig.add_subplot(2,2,3)
ax3.plot(tSkewnessVals,skewnessVals,color="black")
ax3.set_xlabel("time",fontsize=ftSize)
ax3.set_ylabel("skewness",fontsize=ftSize)
ax3.set_title("variation of skewness",fontsize=ftSize)
tSkewnessEnd=datetime.now()
print("Skewness time: ",tSkewnessEnd-tSkewnessStart)

tOnePlotStart=datetime.now()
qPlot=int(Q/2)
tPlot=dt*qPlot

ax4=fig.add_subplot(2,2,4)

ax4.plot(sites[:truncation],strVec2ComplexVecAbs(inData.iloc[qPlot,])[:truncation],color="black")
ax4.set_xlabel("sites",fontsize=ftSize)
ax4.set_ylabel("$|\psi_{n}|$",fontsize=ftSize)
ax4.set_title("Shape of soliton at t = "+str(round(tPlot,3)),fontsize=ftSize)

tOnePlotEnd=datetime.now()
print("one plot time: ",tOnePlotEnd-tOnePlotStart)








fig.suptitle("Sawtooth solution",fontsize=ftSize)
plt.savefig(inDir+"evolution.png")
plt.close()