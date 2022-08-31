from changingNonlinearityFuncs import *


########evolution
tCalcStart=datetime.now()
psiAll=[initVec]
for q in range(0,Q):
    psiCurr=psiAll[q]
    psiNext=oneStepRK4(psiCurr,q)
    psiAll.append(psiNext)
tCalcEnd=datetime.now()
print("calc time: ",tCalcEnd-tCalcStart)

outDir="./changingNonlinearity/0s"+str(s)+"Gmax"+str(Gmax)+"tTot"+str(tTot)+"k0"+str(k0)+"a1"+str(a1)+"coefPi"+str(coefPi)+"/"
pltOut=outDir+"out/"
tValsAll=[q*dt for q in range(0,Q+1)]


######plotting

Path(pltOut).mkdir(parents=True,exist_ok=True)
initWidth=2*np.log(2+np.sqrt(3))/np.abs(a1)
initv=s/a1*(np.exp(a1)-np.exp(-a1))*np.sin(k0+b1)
outTxtName="summary.txt"
with open(outDir+outTxtName,"w") as fPtr:
    fPtr.write("a1 = "+str(a1)+"\n")
    fPtr.write("initial width = "+str(initWidth)+"\n")
    fPtr.write("initial velocity = "+str(initv)+"\n")
    fPtr.write("total time = "+str(tTot)+"\n")
    fPtr.write("pi coef = "+str(coefPi)+"\n")
fPtr.close()


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
    GCurr=round(Gn(tCurr),3)
    FCurr=round(Fn(tCurr),3)
    plt.title("t="+str(round(dt*q,3))+", A1="+str(A1Curr)+", A2="+str(A2Curr)+", B1="+str(B1Curr)+", B2="+str(B2Curr)+", F="+str(FCurr)+", G="+str(GCurr))
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


#plot drift

# Path(outDir).mkdir(parents=True,exist_ok=True)
plt.figure()
plt.plot(tValsAll,drift,color="black")
plt.xlabel("time")
plt.ylabel("drift")
plt.title("Gmax="+str(Gmax)+", tTot="+str(tTot)+", drift="+str(round(drift[-1],3)))
plt.savefig(outDir+"Gmax="+str(Gmax)+", tTot="+str(tTot)+".png")
plt.close()

#plot sigma
plt.figure()
plt.plot(tValsAll,sigmaAll,color="black")
plt.xlabel("time")
plt.ylabel("$\sigma$")
plt.title("Gmax="+str(Gmax)+", tTot="+str(tTot))
plt.savefig(outDir+"Gmax="+str(Gmax)+", tTot="+str(tTot)+"Var.png")
plt.close()

#plot height
plt.figure()
plt.plot(tValsAll,heigtsAll,color="black")
plt.xlabel("time")
plt.ylabel("height")
plt.title("Gmax="+str(Gmax)+", tTot="+str(tTot))
plt.savefig(outDir+"Gmax="+str(Gmax)+", tTot="+str(tTot)+"height.png")
plt.close()
#plot norm
plt.figure()
plt.plot(tValsAll,normsAll,color="black")
plt.xlabel("time")
plt.ylabel("norm")
plt.title("Gmax="+str(Gmax)+", tTot="+str(tTot))
plt.savefig(outDir+"Gmax="+str(Gmax)+", tTot="+str(tTot)+"norm.png")
tPosEnd=datetime.now()

print("calc pos time: ",tPosEnd-tPosStart)
plt.close()