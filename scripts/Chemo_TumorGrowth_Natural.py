# Combination Chemotherapy Optimization with Discrete Dosing
# Temitayo Ajayi, Seyedmohammadhossein Hosseinian, Andrew J. Schaefer, Clifton D. Fuller
#
# ___________________________
# Tumor natural growth models
# ___________________________



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


def simulate_tumor_former(simLength, eulerH, initPops, popMaxes, gompParam, popTypes):
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
    P = [[0. for s in stageSteps] for q in popTypes]
    for q in popTypes:
        P[q][0] = initPops[q]
        for s in stageSteps[1:]:
            P[q][s] = P[q][s-1] + eulerH*(gompParam*(popMaxes[q] - P[q][s-1]))
    return P 


#


def simulate_tumor_latter(simLength, eulerH, initPops, popMaxes, gompParam, popTypes):
    timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
    P = [[0. for s in stageSteps] for q in popTypes]
    for q in popTypes:
        P[q][0] = initPops[q]        
    for s in stageSteps[1:]:
        for q in popTypes:
            P[q][s] = P[q][s-1] + eulerH*(gompParam*(popMaxes[q] - math.log(sum(math.exp(P[g][s-1]) for g in popTypes))))
    return P 


#


# Main
simLength = 10e3                              # in days
#simLength = 7*30
eulerH = 1.0/(8*3.0)                          # as a fraction of a day (e.g., 1/24 for one hour)
popTypes = range(0,4)
#
gompParam = 0.0007
initPops = [19.34, 19.34, 19.34, 19.34]       # Balanced tumor (in log scale)
#initPops = [20.49, 17.95, 17.95, 17.95]      # Imbalanced tumor (in log scale)                
popMaxes_former = [7.0 + ele for ele in initPops]
popMaxes_latter = [27.73 for ele in initPops]
#
timeSteps, stageSteps = det_TimeSteps(simLength, eulerH)
P_former = simulate_tumor_former(simLength, eulerH, initPops, popMaxes_former, gompParam, popTypes)
P_latter = simulate_tumor_latter(simLength, eulerH, initPops, popMaxes_latter, gompParam, popTypes)
#
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

