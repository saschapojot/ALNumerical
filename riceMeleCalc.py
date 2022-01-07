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

####calc position expectation value
tPosStart=datetime.now()
posAll=[]
for psiTmp in psiAll:
    posAll.append(avgPos(psiTmp))

plt.figure()
plt.plot(range(0,Q+1),posAll,color="black")
plt.savefig("./pumping.png")
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