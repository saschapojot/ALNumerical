from mpRK4Funcs import *
import numpy as np

# pert=[np.exp(-n**2) for n in nRange]

initVec=[psi2Soliton(n,0) for n in nRange]

psiAll=[initVec]
for q in range(0,Q):
    print("curr time: "+str(q))
    psiCurr=psiAll[q]
    psiNext=oneStepRK4(psiCurr,q)
    psiAll.append(psiNext)

for q in range(0,Q+1):
    plt.figure()
    psiCurr=psiAll[q]
    psiCurrAbs=[mpmath.fabs(elem) for elem in psiCurr]
    plt.plot(nRange,psiCurrAbs,color="black")
    plt.title(str(q))
    plt.savefig("./out/"+str(q)+".png")
    plt.close()




trueVecsAll=[]
for q in range(0,Q+1):
    tq=dt*q
    yq=[psi2Soliton(n,tq) for n in nRange]
    trueVecsAll.append(yq)

dist=[]
for q in range(0,Q+1):
    diffTmp=np.array(psiAll[q])-np.array(trueVecsAll[q])
    dist.append(np.linalg.norm(diffTmp,ord=2))

plt.figure()
plt.plot(range(0,Q+1),dist,color="black")
plt.savefig("tmp.png")
plt.close()