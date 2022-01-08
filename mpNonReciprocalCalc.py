from mpNonReciprocalFuncs import *


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
tPltStart=datetime.now()
for q in range(0,Q+1):
    plt.figure()
    psiCurr=psiAll[q]
    absTmp=[mpmath.fabs(elem) for elem in psiCurr]
    plt.plot(nRange,absTmp,color="black")
    plt.title(str(q))
    plt.savefig("./out/"+str(q)+".png")
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
outDir="./nonreciprocal/t1"+str(t1Val)+"t2"+str(t2Val)+"/"
Path(outDir).mkdir(parents=True,exist_ok=True)
plt.figure()
plt.plot(range(0,Q+1),drift,color="black")
plt.xlabel("tTot="+str(tTot))
plt.ylabel("drift")
plt.title("$t_{1}=$"+str(t1Val)+", $t_{2}=$"+str(t2Val)+", drift="+str(round(drift[-1],3)))
plt.savefig(outDir+"t1"+str(t1Val)+"t2"+str(t2Val)+".png")
plt.close()
#plot sigma
plt.figure()
plt.plot(range(0,Q+1),sigmaAll,color="black")
plt.xlabel("tTot="+str(tTot))
plt.ylabel("$\sigma$")
plt.title("$t_{1}=$"+str(t1Val)+", $t_{2}=$"+str(t2Val))
plt.savefig(outDir+"t1"+str(t1Val)+"t2"+str(t2Val)+"Var.png")
plt.close()
tPosEnd=datetime.now()
#plot height
plt.figure()
plt.plot(range(0,Q+1),heigtsAll,color="black")
plt.xlabel("tTot="+str(tTot))
plt.ylabel("height")
plt.title("$t_{1}=$"+str(t1Val)+", $t_{2}=$"+str(t2Val))
plt.savefig(outDir+"t1"+str(t1Val)+"t2"+str(t2Val)+"height.png")
#plot norm
plt.figure()
plt.plot(range(0,Q+1),normsAll,color="black")
plt.xlabel("tTot="+str(tTot))
plt.ylabel("norm")
plt.title("$t_{1}=$"+str(t1Val)+", $t_{2}=$"+str(t2Val))
plt.savefig(outDir+"t1"+str(t1Val)+"t2"+str(t2Val)+"norm.png")
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
#     diffTmp=mpmath.array(psiAll[q])-mpmath.array(trueVecsAll[q])
#     dist.append(mpmath.linalg.norm(diffTmp,ord=2))
# tDiffEnd=datetime.now()
# print("diff time: ",tDiffEnd-tDiffStart)
# plt.figure()
# plt.plot(range(0,Q+1),dist,color="black")
# plt.savefig("tmp.png")
# plt.close()