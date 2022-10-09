from checkingSawtoothFuncs import *

########evolution
tCalcStart=datetime.now()
psiAll=[initVec]
for q in range(0,Q):
    psiCurr=psiAll[q]
    psiNext=oneStepRK4(psiCurr,q)
    psiAll.append(psiNext)
tCalcEnd=datetime.now()
print("calc time: ",tCalcEnd-tCalcStart)

outDir="./checkingSawtooth/"+"coef"+str(coefPi)+"/"+"0s"+str(s)+"tTot"+str(tTot)+"a1"+str(a1)+"coefPi"+str(coefPi)+"Q"+str(Q)+"/"
pltOut=outDir+"out/"
Path(pltOut).mkdir(parents=True,exist_ok=True)
tValsAll=[q*dt for q in range(0,Q+1)]

tExactStart=datetime.now()
exactSolutions=[]
for tq in tValsAll:
    exactSolutions.append(exactVec(tq))
tExactEnd=datetime.now()
print("exact time: ",tCalcEnd-tExactStart)

diffNorm=[]
for q in range(0,Q+1):
    diffNorm.append(np.linalg.norm(psiAll[q]-exactSolutions[q],ord=2))


plt.figure()
plt.plot(tValsAll,diffNorm,color="black")

plt.savefig(outDir+"diff.png")
plt.close()

#####start plotting
NPics=1000
sep=int(np.floor(Q/NPics))
tPltStart=datetime.now()
for q in range(0,Q+1):
    if sep==0:
        sep=2
    if q%sep!=0:
        continue
    plt.figure()
    psiCurr=psiAll[q]
    absTmp=[np.abs(elem) for elem in psiCurr]
    plt.plot(nRange,absTmp,color="black")
    tCurr=q*dt
    A1Curr=round(A1(tCurr),3)
    A2Curr=round(A2(tCurr),3)
    B1Curr=round(B1(tCurr),3)
    B2Curr=round(B2(tCurr),3)
    plt.title("t="+str(round(dt*q,3))+", A1="+str(A1Curr)+", A2="+str(A2Curr)+", B1="+str(B1Curr)+", B2="+str(B2Curr))
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
    sgmTmp=np.sqrt(np.abs(position2All[n]-positionAll[n]**2))
    sigmaAll.append(sgmTmp)


#height
heigtsAll=[]
for psiTmp in psiAll:
    heigtsAll.append(height(psiTmp))

#norm
normsAll=[]
for psiTmp in psiAll:
    normsAll.append(norm2(psiTmp))


#summary
outTxtName="summary.txt"
initv=1/a1*(np.exp(a1)-np.exp(-a1))*np.sin(k0+b1)
with open(outDir+outTxtName,"w") as fPtr:
    fPtr.write("dt = "+str(dt))
    fPtr.write("a1 = "+str(a1)+"\n")
    fPtr.write("initial width = "+str(sigmaAll[0])+"\n")
    fPtr.write("initial velocity = "+str(initv)+"\n")
    fPtr.write("total time = "+str(tTot)+"\n")
    fPtr.write("pi coef = "+str(coefPi)+"\n")
fPtr.close()


#plot drift

# Path(outDir).mkdir(parents=True,exist_ok=True)
plt.figure()
plt.plot(tValsAll,drift,color="black")
plt.xlabel("time")
plt.ylabel("drift")
plt.title("tTot="+str(tTot)+", drift="+str(round(drift[-1],3)))
plt.savefig(outDir+"tTot="+str(tTot)+".png")
plt.close()

#plot sigma
plt.figure()
plt.plot(tValsAll,sigmaAll,color="black")
plt.xlabel("time")
plt.ylabel("$\sigma$")
plt.title("tTot="+str(tTot))
plt.savefig(outDir+"tTot="+str(tTot)+"Var.png")
plt.close()

#plot height
plt.figure()
plt.plot(tValsAll,heigtsAll,color="black")
plt.xlabel("time")
plt.ylabel("height")
plt.title("tTot="+str(tTot))
plt.savefig(outDir+"tTot="+str(tTot)+"height.png")
plt.close()
#plot norm
plt.figure()
plt.plot(tValsAll,normsAll,color="black")
plt.xlabel("time")
plt.ylabel("norm")
plt.title("tTot="+str(tTot))
plt.savefig(outDir+"tTot="+str(tTot)+"norm.png")
tPosEnd=datetime.now()

print("calc pos time: ",tPosEnd-tPosStart)
plt.close()

#to csv

outDf=pd.DataFrame(data=psiAll)

outDf.to_csv(outDir+"tTot="+str(tTot)+"data.csv",header=False,index=False)
