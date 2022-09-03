from checkingExactFunc import *

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

tExactStart=datetime.now()
procNum=24
pool0=Pool(procNum)
indsAndExact=pool0.map(exactAtt,range(0,Q+1))
tExactEnd=datetime.now()
print("exact time: ",tExactEnd-tExactStart)
sortedindsAndExact=sorted(indsAndExact,key=lambda elem:elem[0])
exactAll=[]
for elem in sortedindsAndExact:
    exactAll.append(elem[1])
outDir="./chExact/"+"coef"+str(coefPi)+"/"+"0s"+str(s)+"Gmax"+str(Gmax)+"tTot"+str(tTot)+"a1"+str(a1)+"coefPi"+str(coefPi)+"/"
pltOut=outDir+"out/"
tValsAll=[q*dt for q in range(0,Q+1)]

diffNorm2=[np.linalg.norm(np.array(psiAll[q])-np.array(exactAll[q]),ord=2) for q in range(0,Q+1)]

######plotting

Path(pltOut).mkdir(parents=True,exist_ok=True)
# initWidth=2*np.log(2+np.sqrt(3))/np.abs(a1)
# initv=s/a1*(np.exp(a1)-np.exp(-a1))*np.sin(k0+b1)
# outTxtName="summary.txt"
# with open(outDir+outTxtName,"w") as fPtr:
#     fPtr.write("a1 = "+str(a1)+"\n")
#     fPtr.write("initial width = "+str(initWidth)+"\n")
#     fPtr.write("initial velocity = "+str(initv)+"\n")
#     fPtr.write("total time = "+str(tTot)+"\n")
#     fPtr.write("pi coef = "+str(coefPi)+"\n")
# fPtr.close()

plt.figure()
plt.plot(range(0,Q+1),diffNorm2,color="blue")
plt.savefig(outDir+"diff.png")
