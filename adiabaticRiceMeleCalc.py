from adiabaticRiceMeleFuncs import *


initVecMp=[y(n,0) for n in nRange]
initVec=[complex(elem) for elem in initVecMp]

########evolution
tCalcStart=datetime.now()
psiAll=[initVec]
for q in range(0,Q):
    psiCurr=psiAll[q]
    psiNext=oneStepRK4(psiCurr,q)
    psiAll.append(psiNext)
tCalcEnd=datetime.now()
print("calc time: ",tCalcEnd-tCalcStart)

outDir="./adiabaticRiceMele/s"+str(s)+"delta"+str(delta)+"Delta"+str(Delta)+"tTot"+str(tTot)+"/"
pltOut=outDir+"out/"
tValsAll=[q*dt for q in range(0,Q+1)]
######plotting

Path(pltOut).mkdir(parents=True,exist_ok=True)
#####start plotting
NPics=1000
sep=int(np.floor(Q/NPics))
tPltStart=datetime.now()
for q in range(0,Q+1):
    if q%sep!=0:
        continue
    plt.figure()
    psiCurr=psiAll[q]
    absTmp=[mpmath.fabs(elem) for elem in psiCurr]
    plt.plot(nRange,absTmp,color="black")
    plt.title("t="+str(round(dt*q,3)))
    plt.xlabel("site")
    plt.ylabel("height")
    plt.savefig(pltOut+str(q)+".png")
    plt.close()

tPltEnd=datetime.now()
print("plotting time: ",tPltEnd-tPltStart)

####calc position expectation value and spread
tPosStart=datetime.now()
#<x>
positionAll=[]
for psiTmp in psiAll:
    positionAll.append(avgPos(psiTmp))
#drift
drift=[elem-positionAll[0] for elem in positionAll]
#<x^{2}>
position2All=[]
for psiTmp in psiAll:
    position2All.append(avgPos2(psiTmp))
#spread
sigmaAll=[]
for n in range(0,len(psiAll)):
    sgmTmp=mpmath.sqrt(mpmath.fabs(position2All[n]-positionAll[n]**2))
    sigmaAll.append(sgmTmp)
#heigh
heigtsAll=[]
for psiTmp in psiAll:
    heigtsAll.append(height(psiTmp))
#norm
normsAll=[]
for psiTmp in psiAll:
    normsAll.append(norm2(psiTmp))

#plot drift

# Path(outDir).mkdir(parents=True,exist_ok=True)
plt.figure()
plt.plot(tValsAll,drift,color="black")
plt.xlabel("time")
plt.ylabel("drift")
plt.title("s="+str(s)+", $\delta=$"+str(delta)+", $\Delta=$"+str(Delta)+", tTot="+str(tTot)+", drift="+str(round(drift[-1],3)))
plt.savefig(outDir+"s="+str(s)+", $\delta=$"+str(delta)+", $\Delta=$"+str(Delta)+", tTot="+str(tTot)+".png")
plt.close()
#plot sigma
plt.figure()
plt.plot(tValsAll,sigmaAll,color="black")
plt.xlabel("time")
plt.ylabel("$\sigma$")
plt.title("s="+str(s)+", $\delta=$"+str(delta)+", $\Delta=$"+str(Delta)+", tTot="+str(tTot))
plt.savefig(outDir+"s="+str(s)+", $\delta=$"+str(delta)+", $\Delta=$"+str(Delta)+", tTot="+str(tTot)+"Var.png")
plt.close()
tPosEnd=datetime.now()
#plot height
plt.figure()
plt.plot(tValsAll,heigtsAll,color="black")
plt.xlabel("time")
plt.ylabel("height")
plt.title("s="+str(s)+", $\delta=$"+str(delta)+", $\Delta=$"+str(Delta)+", tTot="+str(tTot))
plt.savefig(outDir+"s="+str(s)+", $\delta=$"+str(delta)+", $\Delta=$"+str(Delta)+", tTot="+str(tTot)+"height.png")
#plot norm
plt.figure()
plt.plot(tValsAll,normsAll,color="black")
plt.xlabel("time")
plt.ylabel("norm")
plt.title("s="+str(s)+", $\delta=$"+str(delta)+", $\Delta=$"+str(Delta)+", tTot="+str(tTot))
plt.savefig(outDir+"s="+str(s)+", $\delta=$"+str(delta)+", $\Delta=$"+str(Delta)+", tTot="+str(tTot)+"norm.png")
print("calc pos time: ",tPosEnd-tPosStart)
#####calc diff
# tDiffStart=datetime.now()
# trueVecsAll=[]
# for q in range(0,Q+1):
#     tq=dt*q
#     yqMp=[y(n,tq) for n in nRange]
#     yq=[complex(elem) for elem in yqMp]
#     trueVecsAll.append(yq)
#
# dist=[]
# for q in range(0,Q+1):
#     diffTmp=np.array(psiAll[q])-np.array(trueVecsAll[q])
#     dist.append(np.linalg.norm(diffTmp,ord=2))
# tDiffEnd=datetime.now()
# print("diff time: ",tDiffEnd-tDiffStart)
# plt.figure()
# plt.plot(range(0,Q+1),dist,color="black")
# plt.savefig("tmp.png")
# plt.close()