from cycleFuncs import *

# initVecmp=[y(n,0) for n in nRange]
# initVec=[complex(elem) for elem in initVecmp]

########evolution
tCalcStart=datetime.now()
psiAll=[initVec]
for q in range(0,Q):
    psiCurr=psiAll[q]
    psiNext=oneStepRK4(psiCurr,q)
    psiAll.append(psiNext)
tCalcEnd=datetime.now()
print("calc time: ",tCalcEnd-tCalcStart)

outDir="./cycle/s"+str(s)+"Gmax"+str(Gmax)+"tTot"+str(tTot)+"k0"+str(k0)+"/"
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
    absTmp=[np.abs(elem) for elem in psiCurr]
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