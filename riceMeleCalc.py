from riceMeleFuncs import *

initVec=[y(n,0) for n in nRange]
########evolution
tCalcStart=datetime.now()
psiAll=[initVec]
for q in range(0,Q):
    psiCurr=psiAll[q]
    psiNext=oneStepRK4(psiCurr,q)
    psiAll.append(psiNext)
tCalcEnd=datetime.now()
print("calc time: ",tCalcEnd-tCalcStart)

######plotting
# tPltStart=datetime.now()
# for q in range(0,Q+1):
#     plt.figure()
#     psiCurr=psiAll[q]
#     plt.plot(nRange,np.abs(psiCurr),color="black")
#     plt.title(str(q))
#     plt.savefig("./out/"+str(q)+".png")
#     plt.close()
#
# tPltEnd=datetime.now()
# print("plotting time: ",tPltEnd-tPltStart)

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
    sgmTmp=np.sqrt(np.abs(position2All[n]-positionAll[n]**2))
    sigmaAll.append(sgmTmp)
#plot drift
plt.figure()
plt.plot(range(0,Q+1),drift,color="black")
plt.xlabel("tTot="+str(tTot))
plt.ylabel("drift")
plt.title("$\Delta_{0}=$"+str(Delta0)+", $\delta_{0}=$"+str(delta0)+", $J=$"+str(J)+", $T=$"+str(T)+", $\omega_{F}=$"+str(omegaF)+", $\Gamma=$"+str(Gamma)+", drift="+str(drift[-1]))
plt.savefig("./riceMele/pumpingDelta0"+str(Delta0)+"delta0"+str(delta0)+"J"+str(J)+"T"+str(T)+"omegaF"+str(omegaF)+"Gamma"+str(Gamma)+".png")
plt.close()
#plot sigma
plt.figure()
plt.plot(range(0,Q+1),sigmaAll,color="black")
plt.xlabel("tTot="+str(tTot))
plt.ylabel("$\sigma$")
plt.title("$\Delta_{0}=$"+str(Delta0)+", $\delta_{0}=$"+str(delta0)+", $J=$"+str(J)+", $T=$"+str(T)+", $\omega_{F}=$"+str(omegaF)+", $\Gamma=$"+str(Gamma))
plt.savefig("./riceMele/pumpingDelta0"+str(Delta0)+"delta0"+str(delta0)+"J"+str(J)+"T"+str(T)+"omegaF"+str(omegaF)+"Gamma"+str(Gamma)+"Var.png")
plt.close()
tPosEnd=datetime.now()
print("calc pos time: ",tPosEnd-tPosStart)
#####calc diff
# tDiffStart=datetime.now()
# trueVecsAll=[]
# for q in range(0,Q+1):
#     tq=dt*q
#     yq=[y(n,tq) for n in nRange]
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