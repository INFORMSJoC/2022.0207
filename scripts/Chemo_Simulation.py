# Combination Chemotherapy Optimization with Discrete Dosing
# Temitayo Ajayi, Seyedmohammadhossein Hosseinian, Andrew J. Schaefer, Clifton D. Fuller
#
# ________________________________________
# Simulation of regularized treatment plan
# ________________________________________



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


#Etoposide administration

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


def simulate_tumor(E, simLength, eulerH, initPops, popMaxes, effectCoeffs, effectExpons, gompParam, popTypes, drugTypes):
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
    P = [[0. for s in stageSteps] for q in popTypes]
    for q in popTypes:
        P[q][0] = initPops[q]
        for s in stageSteps[1:]:
            P[q][s] = P[q][s-1] + eulerH*(gompParam*(popMaxes[q] - P[q][s-1])\
                                          - sum(effectCoeffs[d][q]*math.exp(-effectExpons[d][q]*timeSteps[s-1])*E[d][s-1] for d in drugTypes))
    return P 


#


def simulate_WBC(C, simLength, eulerH, drugTypes, wbcInitial, wbcTurnover, wbcProduction, wbcEffect, wbcDelay, wTimeDisc):
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
    wTime = [s for s in range(len(C[0])) if s%wTimeDisc == 0]
    NW = [None for s in wTime]
    stepDelay = int(wbcDelay/(eulerH*wTimeDisc))
    conc_delay = int(wbcDelay/eulerH)
    numDaysForAvg = 1.0
    wAvgOver = int(numDaysForAvg/eulerH)
    NW[0] = wbcInitial
    for s in range(stepDelay):
        NW[s+1] = NW[s] + wTimeDisc*eulerH*(wbcProduction - wbcTurnover*NW[s])
    for s in range(stepDelay,len(NW) - 1):
        NW[s+1] = NW[s] + wTimeDisc*eulerH*(wbcProduction - wbcTurnover*NW[s] - sum(wbcEffect[d]*((1/wAvgOver)*(sum(C[d][s - conc_delay + l] for l in range(wAvgOver))))*NW[s] for d in drugTypes))
    return NW   


#


# Main
withRest ='yes'
#
simLength = 21                                # in days
eulerH = 1.0/(8*3.0)                          # as a fraction of a day (e.g., 1/24 for one hour)
drugTypes = range(0,3)
popTypes = range(0,4)
#
bioCons = [.6,.2,.8]
vol = 15*10**-3                               # volume (in m^3) for drug concentration
gompParam = 0.0007
initPops = [20.49, 17.95, 17.95, 17.95]       # in log scale        
popMaxes = [7.0 + ele for ele in initPops]
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
## WBC 
wbcInitial = 8.0*10**3                        # For numerical reasons the unit is (billion cell per cubic meter)
wbcTurnover = 0.15
wbcProduction = 1.2*10**3                     # For numerical reasons the unit is (billion cell per cubic meter)
wbcEffect = [cEffect, dEffect, eEffect]       # conservative assumption: kill effect on WBC equal to tumor
wbcDelay = 5  
wbcNeutrophilMin = 2.5*10**3                  # For numerical reasons the unit is (billion cell per cubic meter)
wbcLymphocyteMin = 1.0*10**3                  # For numerical reasons the unit is (billion cell per cubic meter)
wbcTimeDisc = int(1.0/eulerH)
wbcNeutrophil = 0.5
wbcLymphocyte = 0.3
#
timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
daySets, mealSets = det_daySetsAndMeals(simLength, eulerH)
U_Capecitabine = makeU_Capecitabine(simLength, eulerH, withRest)
U_Docetaxel = makeU_Docetaxel(simLength, eulerH)
U_Etoposide = makeU_Etoposide(simLength, eulerH, withRest)
C_Capecitabine, E_Capecitabine = makeCandEgivenU(simLength, eulerH, U_Capecitabine, bioCons[0], vol, effectFloors[0])
C_Docetaxel, E_Docetaxel = makeCandEgivenU(simLength, eulerH, U_Docetaxel, bioCons[1], vol, effectFloors[1])
C_Etoposide, E_Etoposide = makeCandEgivenU(simLength, eulerH, U_Etoposide, bioCons[2], vol, effectFloors[2])
U = [U_Capecitabine, U_Docetaxel, U_Etoposide]
C = [C_Capecitabine, C_Docetaxel, C_Etoposide]
E = [E_Capecitabine, E_Docetaxel, E_Etoposide]
P = simulate_tumor(E, simLength, eulerH, initPops, popMaxes, effectCoeffs, effectExpons, gompParam, popTypes, drugTypes)
NW = simulate_WBC(C, simLength, eulerH, drugTypes, wbcInitial, wbcTurnover, wbcProduction, wbcEffect, wbcDelay, wbcTimeDisc)
# Output
stageSteps_out = stageSteps
timeSteps_out = timeSteps
daySets_out = daySets
mealSets_out = mealSets
P_out = P
U_out = [[1000*U[d][s] for s in stageSteps] for d in drugTypes]         # in mg
C_out = [[1000*C[d][s] for s in stageSteps] for d in drugTypes]         # in mg
E_out = [[1000*E[d][s] for s in stageSteps] for d in drugTypes]         # in mg
wTimeDisc = len(timeSteps)/len(NW)
wTime_out = [timeSteps[s] for s in range(len(timeSteps)) if s%wTimeDisc == 0]
wTime_out.append(wTime_out[-1]+1)
NW_out = NW
NW_out.append(NW_out[-1]) 
wbcNeutrophil_out = [wbcNeutrophil*(ele*10**9) for ele in NW_out]
wbcLymphocyte_out = [wbcLymphocyte*(ele*10**9) for ele in NW_out]
wbcNeutrophilMin_out = [wbcNeutrophilMin*10**9 for ele in NW_out]
wbcLymphocyteMin_out = [wbcLymphocyteMin*10**9 for ele in NW_out]
output = [stageSteps_out, timeSteps_out, daySets_out, mealSets_out, P_out, U_out, C_out, E_out, wTime_out, NW_out, wbcNeutrophil_out, wbcLymphocyte_out, wbcNeutrophilMin_out, wbcLymphocyteMin_out, eulerH, vol]
with open('output.pkl', 'wb') as file:
    pickle.dump(output, file)


# ### Plot

#


with open('output.pkl', 'rb') as file:
    output = pickle.load(file)
#
stageSteps = output[0]
timeSteps = output[1]
daySets = output[2]
mealSets = output[3]
P = output[4]
U = output[5]
C = output[6]
E = output[7]
wTime = output[8]
NW = output[9]
Neutrophil = output[10]
Lymphocyte = output[11]
NeutrophilMin = output[12]
LymphocyteMin = output[13]
eulerH = output[14]
vol = output[15]
#
dNames = ['Capecitabine', 'Docetaxel', 'Etoposide']
wbcNames = ['Neutrophil', 'Clinical bound (Neutrophil)', 'Lymphocyte', 'Clinical bound (Lymphocyte)']
popNames = ['tumor-N', 'tumor-C', 'tumor-D', 'tumor-E', 'tumor-Total']
markerPos = [s for s in range(len(timeSteps)) if s%(3.0/eulerH) == 0]
markerPos.append(len(timeSteps)-1)
myMarkers=['o','d','<','+']
myLinestyles=[(0, (5, 5)), 'dashdot', 'dotted', (0, (3, 5, 1, 5, 1, 5))]
#
for q in range(len(P)):
    plt.plot(timeSteps, P[q], linestyle=myLinestyles[q], marker=myMarkers[q], markevery=markerPos, mfc='none') 
ALL = [math.log(sum(math.exp(P[q][s]) for q in range(len(P)))) for s in range(len(P[0]))]
plt.plot(timeSteps, ALL, linestyle='solid', marker='s', markevery=markerPos)
plt.xticks(range(1+int(max(timeSteps))))
plt.legend(popNames)
plt.title('Tumor Cell Population (Logarithmic Scale)')
plt.xlabel('Time (day)')
plt.ylabel('Ln(cells)')
plt.savefig("logPop.png")
plt.close()
#
localMarkerPos=[[],[],[]]
for d in range(1,len(U)):
    for s in range(len(U[0])):
        if U[d][s] > 10:
            localMarkerPos[d].append(s)
localMarkerType=[None, 's', 'o']
for d in range(len(U)):
    plt.plot(timeSteps, U[d], marker=localMarkerType[d], markevery=localMarkerPos[d], mfc='none')
plt.xticks(range(1+int(max(timeSteps))))
plt.legend(dNames)
plt.title('Drug Administration')
plt.xlabel('Time (day)')
plt.ylabel('Administration (mg)')
plt.savefig("DrugAdmin.png")
plt.close()
#
for d in range(len(C)):
    temp = [vol*C[d][s] for s in range(len(C[0]))]
    plt.plot(timeSteps, temp)
plt.xticks(range(1+int(max(timeSteps))))
plt.legend(dNames)
plt.title('Drug Concentration')
plt.xlabel('Time (day)')
plt.ylabel('Concentration (mg/$\mathscr{V}$)')
plt.savefig("Conc.png")
plt.close()
#
for d in range(len(C)): 
    plt.plot(timeSteps, U[d])
    temp = [vol*C[d][s] for s in range(len(C[0]))]
    plt.plot(timeSteps, temp, linestyle=(0, (1, 1)))   
    plt.xticks(range(1+int(max(timeSteps))))
    plt.legend(['Administration (mg)', 'Concentration (mg/$\mathscr{V}$)'])
    plt.title(dNames[d]+' Administration and Concentration')
    plt.xlabel('Time (day)')
    plt.savefig("C_U_"+str(dNames[d])+".png")
    plt.close()
#
plt.plot(wTime, Neutrophil, linestyle='solid', marker='s', markevery=3, mfc='none')
plt.plot(wTime, NeutrophilMin, linestyle='dashed')
plt.plot(wTime, Lymphocyte, linestyle='solid', marker='o', markevery=3, mfc='none')
plt.plot(wTime, LymphocyteMin, linestyle='dotted')
plt.xticks(range(1+int(max(wTime))))
plt.xlabel('Time (days)')
plt.ylabel('White Blood Cells/$m^3$')
plt.legend(wbcNames)
plt.title('White Blood Cell Population')
plt.savefig('WBC.png')
plt.close()

