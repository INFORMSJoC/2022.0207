# Combination Chemotherapy Optimization with Discrete Dosing
# Temitayo Ajayi, Seyedmohammadhossein Hosseinian, Andrew J. Schaefer, Clifton D. Fuller
#
# _______________________________________________________________________________
# Estimating kill factors based on Partial Response Rate (PRR) in clinical trials
# _______________________________________________________________________________



from gurobipy import *
import math
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pandas as pd
import pickle


#


#  Generating discrete times (timeSteps) and the corresponding indices (stageSteps) 
#  simLength = number of days; eulerH = fraction of a day (e.g., 1/24 for an hour)

def det_TimeSteps(simLength, eulerH):
    timeSteps = np.linspace(0,simLength,math.ceil(simLength/eulerH))
    stageSteps = []
    counter = 0
    for s in timeSteps:
        stageSteps.append(counter)
        counter += 1
    return timeSteps, stageSteps


#


# Determining "gompParam" based on "doublingTime = 5 months= 5*30 days", "PInit = 1*10**9", and "Pmax ~ 1*10**12"

longLength = 25*12*30
interval = 1./1
doublingTime = 5*30
timeSteps, stageSteps = det_TimeSteps(longLength, interval)
etaP = [0 for s in stageSteps]
PInit = 1*10**9
Pmax = math.exp(math.log(PInit) + 7.0)
etaP[0] = PInit
gompParam = (1./doublingTime)*math.log((math.log(Pmax/PInit))/(math.log(Pmax/(2*PInit))))


#


# Capecitabine administration

def makeU_Capecitabine(simLength, eulerH):
    if simLength != 7*18 or int(1/eulerH) != 2:
        print("\n \n*** ERROR! ***\n \n")
    else:
        dose = 1.7*1255*(10**(-3))
        timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
        U = [0. for s in stageSteps]
        for cycle in range(6):
            for counter in range(21*int(1/eulerH)):
                if counter <=27:
                    U[cycle*(21*int(1/eulerH))+counter] = dose
    return U


#


# Docetaxel administration

def makeU_Docetaxel(simLength, eulerH):
    if simLength != 7*3*7 or int(1/eulerH) != 1:
        print("\n \n*** ERROR! ***\n \n")
    else:
        dose = 1.7*100*(10**(-3))
        timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
        U = [0. for s in stageSteps]
        for cycle in range(7):
            for counter in range(21*int(1/eulerH)):
                if counter==0:
                    U[cycle*(21*int(1/eulerH))+counter] = dose
        return U


#


#Etoposide administration

def makeU_Etoposide(simLength, eulerH):
    if simLength != 7*21 or int(1/eulerH) != 1:
        print("\n \n*** ERROR! ***\n \n")
    else:
        dose = 1.7*60*(10**(-3))
        timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
        U = [0. for s in stageSteps]
        for cycle in range(7):
            for counter in range(21*int(1/eulerH)):
                if counter< 10:
                    U[cycle*(21*int(1/eulerH))+counter] = dose
        return U


#


# Drug (effective) concentration

def makeCandEgivenU(simLength, eulerH, U, bioCon, vol, effectFloor, drugName):
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)     
    C = [0 for s in stageSteps]
    E = [0 for s in stageSteps]
    for s in stageSteps[1:]:
        C[s] = C[s-1] - (eulerH * bioCon * C[s-1]) + (U [s-1] / vol)
        E[s] = max(0, C[s] - effectFloor) 
    Cmax = max(C)
    plt.plot(timeSteps, C, label = 'Concen.', marker = 'o')
    plt.plot(timeSteps, E, label = 'E. Concen.')
    plt.xlabel('Time (days)')
    plt.ylabel('Concentration')
    plt.title('Clinical Study, '+drugName+' (Cmax = '+str(round(Cmax, 2))+') gr/m3') 
    plt.savefig('simConcenPlot_'+drugName+'.png')
    plt.close()
    return C, E


#


def makeKillEffectPerturbations(sd, N):
    np.random.seed(1)
    perts = list(np.random.normal(loc = 0, scale = sd, size = N))
    return perts


#


def estimateKillParam(C, E, simLength, eulerH, PInit, Pmax, effectExpon, perts, gompParam, tarEndPop, drugName, devFromHalfPop):
    nTrial = len(perts)
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)  
    model = Model()
    model.setParam('FeasibilityTol', 10**(-8))
    # vars
    P = [[None for s in stageSteps] for n in range(nTrial)]
    for n in range(nTrial):
        for s in stageSteps:
            P[n][s] = model.addVar(lb = -float('infinity'))
        model.addConstr(P[n][0] == PInit)     
    epsl = model.addVar(lb = 0.0)
    M = model.addVar(lb = -float('infinity'))
    R = model.addVar(lb = 0.0)
    # constraints
    for n in range(nTrial):
        for s in stageSteps[1:]:
            model.addConstr(P[n][s] == P[n][s-1] + eulerH*(gompParam*(Pmax - P[n][s-1]) - (R + perts[n])*np.exp(-effectExpon*timeSteps[s-1])*E[s-1])) 
    model.addConstr(M == (1.0/nTrial)*sum(P[n][-1] for n in range(nTrial)))
    model.addConstr(M - tarEndPop - devFromHalfPop <= epsl)
    model.addConstr(tarEndPop + devFromHalfPop - M <= epsl)
    # objective
    model.setObjective(epsl, sense = GRB.MINIMIZE)
    model.optimize()
    for trial in range(len(perts)):
        plt.plot(timeSteps, [P[trial][t].x for t in range(len(P[0]))])
    plt.xlabel('Time (days)')
    plt.title("Avg. end population (log)_"+drugName+" = "+str(M.x)) 
    plt.savefig('simPopPlot_perts_'+drugName+'.png')
    plt.close()
    tempNumber = 0
    for n in range(nTrial):
        if P[n][-1].x <= tarEndPop:
            tempNumber += 1
    partialResponse = tempNumber/nTrial
    simulatedPop = [[0 for s in stageSteps] for n in range(nTrial)]
    for n in range(nTrial):
        for s in stageSteps:
            simulatedPop[n][s] = P[n][s].x
    return R.x, partialResponse, simulatedPop, timeSteps


#


def simulate_by_etaEst(E, simLength, eulerH, PInit, Pmax, effectExpon, gompParam, etaEst, drugName):
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
    etaP = [0 for s in stageSteps]
    etaP[0] = PInit
    for s in stageSteps[1:]:
        etaP[s] = etaP[s-1] + eulerH*(gompParam*(Pmax - etaP[s-1]) - etaEst*np.exp(-effectExpon*timeSteps[s-1])*E[s-1]) 
    plt.plot(timeSteps, [etaP[s] for s in stageSteps])
    plt.xlabel('Time (days)')
    plt.savefig('simPopPlot_etaEst_'+drugName+'.png')
    plt.close()


#


def run_estimateKillParam_Capecitabine(simLength, eulerH, sd, numTrials, drugName, devFromHalfPop):
    perts = makeKillEffectPerturbations(sd, numTrials)
    #
    fractionGoal = 0.1
    gompParam = 0.0007
    PInit = math.log(1*10**9)
    Pmax = PInit + 7.0
    effectExpon = 0.0
    tarEndPop = math.log(fractionGoal*math.exp(PInit))
    bioCon = 0.6
    vol = 15*10**(-3)
    effectFloor = 0.0
    U = makeU_Capecitabine(simLength, eulerH)
    C, E = makeCandEgivenU(simLength, eulerH, U, bioCon, vol, effectFloor, drugName)   
    etaEst, partialResponse, simulatedPop, timeSteps = estimateKillParam(C, E, simLength, eulerH, PInit, Pmax, effectExpon, perts, gompParam, tarEndPop, drugName, devFromHalfPop)
    simulate_by_etaEst(E, simLength, eulerH, PInit, Pmax, effectExpon, gompParam, etaEst, drugName)
    return etaEst, partialResponse, simulatedPop, timeSteps


#


def run_estimateKillParam_Docetaxel(simLength, eulerH, sd, numTrials, drugName, devFromHalfPop):
    perts = makeKillEffectPerturbations(sd, numTrials)
    #
    fractionGoal = 0.1
    gompParam = 0.0007
    PInit = math.log(1*10**9)
    Pmax = PInit + 7.0
    effectExpon = 0.0
    tarEndPop = math.log(fractionGoal*math.exp(PInit))
    bioCon = 0.2
    vol = 15*10**(-3)
    effectFloor = 0.0
    U = makeU_Docetaxel(simLength, eulerH)
    C, E = makeCandEgivenU(simLength, eulerH, U, bioCon, vol, effectFloor, drugName)   
    etaEst, partialResponse, simulatedPop, timeSteps = estimateKillParam(C, E, simLength, eulerH, PInit, Pmax, effectExpon, perts, gompParam, tarEndPop, drugName, devFromHalfPop)
    simulate_by_etaEst(E, simLength, eulerH, PInit, Pmax, effectExpon, gompParam, etaEst, drugName)
    return etaEst, partialResponse, simulatedPop, timeSteps


#


def run_estimateKillParam_Etoposide(simLength, eulerH, sd, numTrials, drugName, devFromHalfPop):
    perts = makeKillEffectPerturbations(sd, numTrials)
    #
    fractionGoal = 0.1
    gompParam = 0.0007
    PInit = math.log(1*10**9)
    Pmax = PInit + 7.0
    effectExpon = 0.0
    tarEndPop = math.log(fractionGoal*math.exp(PInit))
    bioCon = 0.8
    vol = 15*10**(-3)
    effectFloor = 0.5
    U = makeU_Etoposide(simLength, eulerH)
    C, E = makeCandEgivenU(simLength, eulerH, U, bioCon, vol, effectFloor, drugName)   
    etaEst, partialResponse, simulatedPop, timeSteps = estimateKillParam(C, E, simLength, eulerH, PInit, Pmax, effectExpon, perts, gompParam, tarEndPop, drugName, devFromHalfPop)
    simulate_by_etaEst(E, simLength, eulerH, PInit, Pmax, effectExpon, gompParam, etaEst, drugName)
    return etaEst, partialResponse, simulatedPop, timeSteps


#


etaEst_C, partialResponse_C, simulatedPop_C, timeSteps_C = run_estimateKillParam_Capecitabine(7*18, 1./2, 7*10**-6, 1000, 'Capecitabine',0.14)
with open('timeSteps_Capecitabine.pkl', 'wb') as file:
    pickle.dump(timeSteps_C, file)
with open('simulatedPop_Capecitabine.pkl', 'wb') as file:
    pickle.dump(simulatedPop_C, file)
#    
etaEst_D, partialResponse_D, simulatedPop_D, timeSteps_D = run_estimateKillParam_Docetaxel(7*3*7, 1./1, 8*10**-4, 1000, 'Docetaxel', 0.02)
with open('timeSteps_Docetaxel.pkl', 'wb') as file:
    pickle.dump(timeSteps_D, file)
with open('simulatedPop_Docetaxel.pkl', 'wb') as file:
    pickle.dump(simulatedPop_D, file)
#
etaEst_E, partialResponse_E, simulatedPop_E, timeSteps_E = run_estimateKillParam_Etoposide(7*21, 1./1, 5*10**-4, 1000, 'Etoposide', 0.32)
with open('timeSteps_Etoposide.pkl', 'wb') as file:
    pickle.dump(timeSteps_E, file)
with open('simulatedPop_Etoposide.pkl', 'wb') as file:
    pickle.dump(simulatedPop_E, file)


# ## Plots

#


with open('timeSteps_Capecitabine.pkl', 'rb') as file:
    timeSteps_C = pickle.load(file)
with open('simulatedPop_Capecitabine.pkl', 'rb') as file:
    simulatedPop_C = pickle.load(file)
timeSteps_C = [(1/7.0)*ele for ele in timeSteps_C]
for trial in range(len(simulatedPop_C)):
    plt.plot(timeSteps_C, simulatedPop_C[trial], linewidth=0.5)
plt.plot(timeSteps_C, [math.log(0.1*10**9) for s in timeSteps_C], linewidth=1.5, linestyle='dashed')
plt.xticks(range(1+int(max(timeSteps_C))))
plt.xlabel('Time (weeks)')
plt.ylabel('Ln(cells)')
plt.title('Tumor Cell Population under perturbation of\nCapecitabine kill effect') 
plt.text(0, 18.2, 'PRR = '+str(int(partialResponse_C*100))+'%')
plt.savefig('simulation_Capecitabine.png')
plt.close()
#
with open('timeSteps_Docetaxel.pkl', 'rb') as file:
    timeSteps_D = pickle.load(file)
with open('simulatedPop_Docetaxel.pkl', 'rb') as file:
    simulatedPop_D = pickle.load(file)
timeSteps_D = [(1/7.0)*ele for ele in timeSteps_D]
for trial in range(len(simulatedPop_D)):
    plt.plot(timeSteps_D, simulatedPop_D[trial], linewidth=0.5)
plt.plot(timeSteps_D, [math.log(0.1*10**9) for s in timeSteps_D], linewidth=1.5, linestyle='dashed')
plt.xticks(range(1+int(max(timeSteps_D))))
plt.xlabel('Time (weeks)')
plt.ylabel('Ln(cells)')
plt.title('Tumor Cell Population under perturbation of\nDocetaxel kill effect') 
plt.text(0, 18.2, 'PRR = '+str(int(partialResponse_D*100))+'%')
plt.savefig('simulation_Docetaxel.png')
plt.close()
#
with open('timeSteps_Etoposide.pkl', 'rb') as file:
    timeSteps_E = pickle.load(file)
with open('simulatedPop_Etoposide.pkl', 'rb') as file:
    simulatedPop_E = pickle.load(file)
timeSteps_E = [(1/7.0)*ele for ele in timeSteps_E]
for trial in range(len(simulatedPop_E)):
    plt.plot(timeSteps_E, simulatedPop_E[trial], linewidth=0.5)
plt.plot(timeSteps_E, [math.log(0.1*10**9) for s in timeSteps_E], linewidth=1.5, linestyle='dashed')
plt.xticks(range(1+int(max(timeSteps_E))))
plt.xlabel('Time (weeks)')
plt.ylabel('Ln(cells)')
plt.title('Tumor Cell Population under perturbation of\nEtoposide kill effect') 
plt.text(0, 18.2, 'PRR = '+str(int(partialResponse_E*100))+'%')
plt.savefig('simulation_Etoposide.png')
plt.close()
#
print("\n \n*------------------------------------------------------*")
print("\n*** Kill factor_Capecitabine = ", etaEst_C,"***")
print("\n*** Partial Response_Capecitabine = ", round(partialResponse_C*100,2),"% ***")
print("\n*------------------------------------------------------*")
print("\n*** Kill factor_Docetaxel = ", etaEst_D,"***")
print("\n*** Partial Response_Docetaxel = ", round(partialResponse_D*100,2),"% ***")
print("\n*------------------------------------------------------*")
print("\n*** Kill factor_Etoposide = ", etaEst_E,"***")
print("\n*** Partial Response_Etoposide = ", round(partialResponse_E*100,2),"% ***")
print("\n*------------------------------------------------------*")

