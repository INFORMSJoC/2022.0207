# Combination Chemotherapy Optimization with Discrete Dosing
# Temitayo Ajayi, Seyedmohammadhossein Hosseinian, Andrew J. Schaefer, Clifton D. Fuller
#
# _______________________________
# Tumor growth under chemotherapy
# _______________________________



import math
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pandas as pd


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


# Partitioning the "stageSteps" into days (outputs "daySets") as well as indices for meals (outputs "mealSets")

def det_daySetsAndMeals(simLength, eulerH):
    dayPoints = 1.0/eulerH
    if dayPoints.is_integer():
        daySets = [[int(day*dayPoints + d) for d in range(int(dayPoints))] for day in range(simLength)]
        mealSets = [[daySets[day][0], daySets[day][int(dayPoints/3)],daySets[day][2*int(dayPoints/3)]] for day in range(simLength)]
        return daySets, mealSets
    else:
        print('*** "dayPoints" NOT integral! *** \n \n')
        input('Press any key to continue...')


#


# Capecitabine administration

def makeU_Capecitabine(simLength, eulerH, withRest):
    dose4pill = 4*500*(10**(-3))
    dose3pill = 3*500*(10**(-3))
    dose2pill = 2*500*(10**(-3))
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
    daySets, mealSets = det_daySetsAndMeals(simLength, eulerH)
    U = [0. for s in stageSteps]
    afterDayThree=[]
    for day in range(len(mealSets)):
        if day<=3:
            U[mealSets[day][0]] = dose4pill
            U[mealSets[day][1]] = dose4pill
        else:
            afterDayThree.extend(mealSets[day])
    for meal in range(len(afterDayThree)):
        if meal%2 == 0:
            U[afterDayThree[meal]] = dose3pill
        else:
            U[afterDayThree[meal]] = dose2pill
    if withRest=='yes':
        # One day rest after 16 days
        rest = [int(16*(1/eulerH)+counter+1) for counter in range(int(1/eulerH))]
        for r in rest:
            U[r] = 0.0
    return U


#


# Docetaxel administration

def makeU_Docetaxel(simLength, eulerH):
    dose = [1.7*100*10**(-3), 1.7*75*10**(-3), 1.7*75*10**(-3)]
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
    U = [0. for s in stageSteps]
    for cycle in range(int(simLength/7.)):
        for counter in range(7*int(1/eulerH)):
            if counter==0:
                U[cycle*(7*int(1/eulerH))+counter] = dose[cycle]
    return U


#


# Etoposide administration

def makeU_Etoposide(simLength, eulerH, withRest):
    dose = 1*50*(10**(-3))
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
    daySets, mealSets = det_daySetsAndMeals(simLength, eulerH)
    allMeals = []
    for meal in mealSets:
        allMeals.extend(meal)
    U = [0. for s in stageSteps]
    for meal in range(len(allMeals)):
        if meal%2 == 0:
            U[allMeals[meal]] = dose
    if withRest=='yes':
        # One day rest after 16 days
        rest = [int(16*(1/eulerH)+counter+1) for counter in range(int(1/eulerH))]
        for r in rest:
            U[r] = 0.0
    return U


#


# Drug (effective) concentration

def makeCandEgivenU(simLength, eulerH, U, bioCon, vol, effectFloor):
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)     
    C = [0 for s in stageSteps]
    E = [0 for s in stageSteps]
    for s in stageSteps[1:]:
        C[s] = C[s-1] - (eulerH * bioCon * C[s-1]) + (U [s-1] / vol)
        E[s] = max(0, C[s] - effectFloor) 
    return C, E


#


def simulate_tumor_former(E, simLength, eulerH, initPops, popMaxes, effectCoeffs, effectExpons, gompParam, popTypes, drugTypes):
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
    P = [[0. for s in stageSteps] for q in popTypes]
    for q in popTypes:
        P[q][0] = initPops[q]
        for s in stageSteps[1:]:
            P[q][s] = P[q][s-1] + eulerH*(gompParam*(popMaxes[q] - P[q][s-1])\
                                          - sum(effectCoeffs[d][q]*math.exp(-effectExpons[d][q]*timeSteps[s-1])*E[d][s-1] for d in drugTypes))
    return P 


#


def simulate_tumor_latter(E, simLength, eulerH, initPops, popMaxes, effectCoeffs, effectExpons, gompParam, popTypes, drugTypes):
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
    P = [[0. for s in stageSteps] for q in popTypes]
    for q in popTypes:
        P[q][0] = initPops[q]
    for s in stageSteps[1:]:
        for q in popTypes:
            P[q][s] = P[q][s-1] + eulerH*(gompParam*(popMaxes[q] - math.log(sum(math.exp(P[g][s-1]) for g in popTypes)))\
                                          - sum(effectCoeffs[d][q]*math.exp(-effectExpons[d][q]*timeSteps[s-1])*E[d][s-1] for d in drugTypes))
    return P 


#


# Main
withRest='yes'

simLength = 21                                # in days
eulerH = 1.0/(8*3.0)                          # as a fraction of a day (e.g., 1/24 for one hour)
drugTypes = range(0,3)
popTypes = range(0,4)
#
bioCons = [.6,.2,.8]
vol = 15*10**-3                               # volume (in m^3) for drug concentration
gompParam = 0.0007
initPops = [20.49, 17.95, 17.95, 17.95]       # in log scale        
popMaxes_former = [7.0 + ele for ele in initPops]
popMaxes_latter = [27.73 for ele in initPops]
resistEffect = [0.25, 0.25, 0.25]
cEffect = 7.2*10**(-5)                       # for Capecitabine
dEffect = 8.0*10**(-3)                       # for Docetaxel
eEffect = 5.1*10**(-3)                       # for Etoposide
effectCoeffs = [[cEffect, resistEffect[0]*cEffect, cEffect, cEffect], 
                [dEffect, dEffect, resistEffect[1]*dEffect, dEffect], 
                [eEffect, eEffect, eEffect, resistEffect[2]*eEffect]]
weeklyEffectExpons = [[.04, .04, .04, .04], [.0876, .0876, .0876, .0876], [.1, .1, .1, .1]] 
effectExpons = [[(1/7.0)*weeklyEffectExpons[d][q] for q in popTypes] for d in drugTypes]
# Operational parameters
effectFloors = [.0, .0, .5]
#
timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
daySets, mealSets = det_daySetsAndMeals(simLength, eulerH)
#
U_Capecitabine = makeU_Capecitabine(simLength, eulerH, withRest)
U_Docetaxel = makeU_Docetaxel(simLength, eulerH)
U_Etoposide = makeU_Etoposide(simLength, eulerH, withRest)
C_Capecitabine, E_Capecitabine = makeCandEgivenU(simLength, eulerH, U_Capecitabine, bioCons[0], vol, effectFloors[0])
C_Docetaxel, E_Docetaxel = makeCandEgivenU(simLength, eulerH, U_Docetaxel, bioCons[1], vol, effectFloors[1])
C_Etoposide, E_Etoposide = makeCandEgivenU(simLength, eulerH, U_Etoposide, bioCons[2], vol, effectFloors[2])
U = [U_Capecitabine, U_Docetaxel, U_Etoposide]
C = [C_Capecitabine, C_Docetaxel, C_Etoposide]
E = [E_Capecitabine, E_Docetaxel, E_Etoposide]
P_former = simulate_tumor_former(E, simLength, eulerH, initPops, popMaxes_former, effectCoeffs, effectExpons, gompParam, popTypes, drugTypes)
P_latter = simulate_tumor_latter(E, simLength, eulerH, initPops, popMaxes_latter, effectCoeffs, effectExpons, gompParam, popTypes, drugTypes)
for cellType in popTypes:
    N_former = [math.exp(P_former[cellType][i]) for i in range(len(P_former[cellType]))]
    N_latter = [math.exp(P_latter[cellType][i]) for i in range(len(P_latter[cellType]))]
    #
    plt.plot(timeSteps, N_former)
    plt.plot(timeSteps, N_latter, linestyle='dashed')
    plt.xlabel('Time (day)')
    plt.ylabel('Cell Population')
    plt.savefig("cellType_"+str(cellType)+".png")
    plt.close()
    print('cellType = '+str(cellType))
    print('N_E1 = '+str(N_latter[-1]))
    print('N_E2 = '+str(N_former[-1]))
    diff = (N_latter[-1] - N_former[-1])
    per = (N_latter[-1] - N_former[-1])/N_latter[-1]
    print('Difference in cell count = '+str(diff)) 
    print('Difference ratio = '+str(per)) 
    print("-------------------------------------")

