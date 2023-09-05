#  ___________________________________________________________________________________________________________
# |                                                                                                           |
# |                       Combination Chemotherapy Optimization with Discrete Dosing                          |
# |                                                                                                           |
# |           Temitayo Ajayi, Seyedmohammadhossein Hosseinian, Andrew J. Schaefer, Clifton D. Fuller          |
# |___________________________________________________________________________________________________________|


# # Stochastic model with a probability-based objective function

#


from gurobipy import *
import math
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pandas as pd


# ### Discretization setup

#


#  Generating discrete times (timeSteps) and the corresponding indices (stageSteps) 
#  treatmentLength = number of days; eulerH = fraction of a day (e.g., 1/24 for an hour)

def gen_TimeSteps(treatmentLength, eulerH):
    timeSteps = np.linspace(0,treatmentLength,math.ceil(treatmentLength/eulerH))
    stageSteps = []
    counter = 0
    for s in timeSteps:
        stageSteps.append(counter)
        counter += 1
    return timeSteps, stageSteps


#


# Partitioning the "stageSteps" into days (outputs "daySets") as well as indices for meals (outputs "mealSets")

def gen_daySetsAndMeals(treatmentLength, eulerH):
    dayPoints = 1.0/eulerH
    if dayPoints.is_integer():
        daySets = [[int(day*dayPoints + d) for d in range(int(dayPoints))] for day in range(treatmentLength)]
        mealSets = [[daySets[day][0], daySets[day][int(dayPoints/3)],daySets[day][2*int(dayPoints/3)]] for day in range(treatmentLength)]
        return daySets, mealSets
    else:
        print('*** "dayPoints" NOT integral! *** \n \n')
        input('Press any key to continue...')


# ### ODE variables and constraints

#


# Cell population variables

def sto_PopulationVars(model, scenarios, popTypes, stageSteps):
    P = [[[None for s in stageSteps] for q in popTypes] for e in scenarios]
    for e in scenarios:
        for q in popTypes:
            for s in stageSteps:
                P[e][q][s] = model.addVar(lb = -float("inf"), ub = float("inf"), name = "P_"+str(e)+"_"+str(q)+"_"+str(s))
    return P


#


# Drug administration variables

def gen_DrugAdminVars(model, drugTypes, stageSteps): 
    U = [[None for s in stageSteps] for d in drugTypes]
    for d in drugTypes:
        for s in stageSteps:
            U[d][s] = model.addVar(lb = 0.0, ub = float("inf"), name = "U_"+str(d)+"_"+str(s))
    return U


#


# Drug concentration variables

def gen_DrugConcenVars(model, drugTypes, stageSteps):
    C = [[None for s in stageSteps] for d in drugTypes]
    for d in drugTypes:
        for s in stageSteps:
            C[d][s] = model.addVar(lb = 0.0, ub = float("inf"), name = "C_"+str(d)+"_"+str(s))
    return C


#


# Drug effective-concentration variables (effective concentration of the drugs depends on the cell types)

def gen_DrugEffectVars(model, drugTypes, stageSteps):
    E = [[None for s in stageSteps] for d in drugTypes]
    for d in drugTypes:
        for s in stageSteps:
            E[d][s] = model.addVar(lb = 0.0, ub = float("inf"), name = "E_"+str(d)+"_"+str(s))
    return E


#


# Scenario binary variables

def sto_ScenarioVars(model, scenarios):
    ZScenario = [None for e in scenarios]
    for e in scenarios:
        ZScenario[e] = model.addVar(vtype = GRB.BINARY, name = 'ZScenario_'+str(e))
    return ZScenario


#


# ODE discretization: drug concentration

def gen_ConcenODEConstraints(model, C, U, eulerH, bioCons, vol):
    drugTypes = range(len(U))
    stageStepsLessOne = range(len(U[0]) - 1)
    for d in drugTypes:
        model.addConstr(C[d][0] == 0)
        for s in stageStepsLessOne:
            model.addConstr(C[d][s+1] == C[d][s] - (eulerH * bioCons[d] * C[d][s]) + (U[d][s] / vol))


#


# ODE discretization: tumor dynamics

def sto_PopODEConstraints(model, P, E, eulerH, gompParam, popMaxes, effectCoeffs, effectExpons, initPops, timeSteps):
    scenarios = range(len(P))
    popTypes = range(len(P[0]))
    drugTypes = range(len(E))
    stageStepsLessOne = range(len(E[0]) - 1)
    for e in scenarios:    
        for q in popTypes:
            model.addConstr(P[e][q][0] == initPops[e][q])
            for s in stageStepsLessOne:
                model.addConstr(P[e][q][s + 1] == P[e][q][s] + eulerH*(gompParam*(popMaxes[e][q] - P[e][q][s])\
                - sum(effectCoeffs[d][q]*math.exp(-effectExpons[d][q]*timeSteps[s])*E[d][s] for d in drugTypes)))


#


# Chance constraints

def sto_chanceConstraints(model, alpha, P, ZScenario, popMaxes, initPops, surgerySize, scenarioProb):
    scenarios = range(len(P))
    popTypes = range(len(P[0]))
    lastStage = len(P[0][0]) - 1
    numerPop = [[math.exp(initPops[e][q]) for q in popTypes] for e in scenarios]
    denomSumPop = [sum(math.exp(initPops[e][q]) for q in popTypes) for e in scenarios]
    constTerm = [[math.log(numerPop[e][q]/denomSumPop[e]) for q in popTypes] for e in scenarios]     
    for e in scenarios:
        for q in popTypes:
            model.addConstr(P[e][q][lastStage] <= popMaxes[e][q]*(1 - ZScenario[e]) + constTerm[e][q] + surgerySize)
    model.addConstr(sum(scenarioProb[e]*ZScenario[e] for e in scenarios) >= alpha)


# ### Operational constraints (and variables)

#


# Max dose per time step

def gen_MaxDoseConstraints(model, U, opParams):
    if 'doseCaps' in opParams:
        doseCaps = opParams['doseCaps']
        for d in range(len(U)):
            for s in range(len(U[0])):
                model.addConstr(U[d][s] <= doseCaps[d])
    else:
        print('\n \n *** "doseCaps" NOT included! *** \n \n')
        input('Press any key to continue...')


#


# Max daily dose

def gen_MaxCumDailyDoseConstraints(model, U, opParams, daySets):
    if 'dailyDoseCaps' in opParams:
        dailyDoseCaps = opParams['dailyDoseCaps']
        for d in range(len(U)):
            for day in daySets:
                model.addConstr(sum(U[d][s] for s in day) <= dailyDoseCaps[d])
    else:
        print('*** "dailyDoseCaps" NOT included! *** \n \n')
        input('Press any key to continue...')


#


# Max dose per time step based on infusion rate 

def gen_InfusionRateConstraints(model, U, eulerH, opParams):
    if 'infusionRates' in opParams:
        infusionRates = opParams['infusionRates']
        # The parameter "eqHR" gives how many hours eulerH is
        eqHR = eulerH*24.0
        for d in range(len(U)):
            for s in range(len(U[0])):
                model.addConstr(U[d][s] <= eqHR*infusionRates[d], name = "infusionRates_"+str(d)+"_"+str(s))
    else:
        print('*** "infusionRates" NOT included! *** \n \n')
        input('Press any key to continue...')


#


#Max concentration

def gen_MaxConcentrationConstraints(model, C, vol, opParams):
    if 'maxConcs' in opParams:
        maxConcs = opParams['maxConcs']
        for d in range(len(C)):
            for s in range(len(C[0])):
                model.addConstr(C[d][s] <= maxConcs[d], name = "maxConcs_"+str(d)+"_"+str(s))
    else:
        print('*** "maxConcs" NOT included! *** \n \n')
        input('Press any key to continue...')


#


#Pills

def gen_PillAdminVarsAndConstraints(model, U, opParams, mealSets, pillRegularize):
    if 'pillSizes' in opParams:
        pillSizes = opParams['pillSizes']        
        allTimeIndices = range(len(U[0]))
        allMeals = []    
        for day in mealSets: 
            allMeals += day
        nonMealTimes = list(set(allTimeIndices).difference(set(allMeals)))    
        ZPill = [[[None for m in range(len(day))] for day in mealSets] for d in range(len(U))]
        for d in range(len(U)):
            if pillSizes[d] >= 0.01:
                dayCount = 0
                for day in mealSets:
                    for m in range(len(day)):
                        ZPill[d][dayCount][m] = model.addVar(lb = 0, ub = float("inf"), vtype = GRB.INTEGER, name = 'Z_Pill_'+str(d)+'_'+str(dayCount)+'_'+str(m)+'_'+str(day[m]))
                        model.addConstr(U[d][day[m]] == ZPill[d][dayCount][m]*pillSizes[d])
                    dayCount +=1
                # Enforcing the same number of pills for each (pill, week, meal), e.g., # Capecitabine pills after breakfast in week 1
                # The default is: pillRegularize = 'no'
                if pillRegularize == 'yes':          
                    numberWeeks = int(len(mealSets)/7)
                    for week in range(numberWeeks):
                        for itr in range(1,7):
                            for m in range(3):
                                model.addConstr(ZPill[d][week*7][m] == ZPill[d][week*7+itr][m])                    
                for s in nonMealTimes:
                    model.addConstr(U[d][s] == 0) 
        return ZPill
    else:
        print('*** "pillSizes" NOT included! *** \n \n')
        input('Press any key to continue...')


#


# Rest days

def gen_RestDaysConstraintsAndVars(model, U, opParams, daySets):
    if 'restDays' in opParams and 'dailyDoseCaps' in opParams:
        restDays = opParams['restDays']
        dailyDoseCaps = opParams['dailyDoseCaps']        
        posRest = list(filter(lambda x: x >= 1, restDays))
        useRest = list(filter(lambda x: restDays[x] in posRest, range(len(restDays))))
        ZRest = [[None for day in daySets] for d in range(len(U))]
        for d in useRest:
            for dayCount in range(len(daySets)):
                ZRest[d][dayCount] = model.addVar(vtype = GRB.BINARY, name = 'ZRest_'+str(d)+'_'+str(dayCount))
        for d in useRest:
            for dayCount in range(len(daySets)):
                model.addConstr(sum(U[d][s] for s in daySets[dayCount]) <= dailyDoseCaps[d]*(1 - ZRest[d][dayCount]))
                model.addConstr(sum(1 - ZRest[d][dayCount + l] for l in range(0, min(restDays[d], len(daySets) - dayCount))) <= 1 )
        return ZRest 
    else:
        print('*** "restDays" (or "dailyDoseCaps") NOT included! *** \n \n')
        input('Press any key to continue...')


#


# Linearization of drug effective-concentration, i.e., E=max{0, C  -\beta_eff}

def gen_RELUEffectiveVarsAndConstraints(model, C, E, opParams):
    if 'useEffectiveConcs' in opParams: 
        useEffectiveConcs = opParams['useEffectiveConcs']
        normalConcs = list(set(range(len(C))).difference(set(useEffectiveConcs)))
        for d in normalConcs:
            for s in range(len(E[0])):
                model.addConstr(E[d][s] == C[d][s])
        if len(useEffectiveConcs) > 0 and 'maxConcs' in opParams and 'effectFloors' in opParams:
            maxConcs = opParams['maxConcs']
            effectFloors = opParams['effectFloors']
            ZEff = [[None for s in range(len(E[0]))] for dE in range(len(useEffectiveConcs))]
            for dE in range(len(useEffectiveConcs)):
                d  =useEffectiveConcs[dE]
                for s in range(len(E[0])):
                    ZEff[dE][s] = model.addVar(vtype = GRB.BINARY, name = 'Z_Eff'+'_'+str(d)+'_'+str(s))
                    model.addConstr(E[d][s] + effectFloors[d] >= C[d][s])
                    model.addConstr(E[d][s] <= maxConcs[d]*ZEff[dE][s]) 
                    model.addConstr(E[d][s] + effectFloors[d] <= C[d][s] + maxConcs[d]*(1 - ZEff[dE][s]))
                    model.addConstr(E[d][s] <= C[d][s])
            return ZEff
    else:
        print('*** "useEffectiveConcs" is missing! Please include this array even if it is empty. *** \n \n')
        input('Press any key to continue...')


# White Blood Cells (WBC)

#


# WBC population variables

def gen_WBCPopVars(model, C, wTimeDisc):
    wTime = [s for s in range(len(C[0])) if s%wTimeDisc == 0]
    NW = [None for s in wTime]
    for s in range(len(wTime)):
        NW[s] = model.addVar(lb = 0.0, ub = float("inf"), name = 'NW_'+str(s))
    return NW


#


# Binary and continuous variables for discrete approximation

def gen_WBCBinaryModelVars(model, C, eulerH, wbcMin, wbcInitial, wTimeDisc):
    wTime = [s for s in range(len(C[0])) if s%wTimeDisc == 0]
    # Compute the state discretization of wbc amount
    wDisc = np.linspace(wbcMin, wbcInitial, 21)
    #V vars are continuous variables, they represent the possible values of (delayed) drug concentration C
    # MC estimates the value of C*NW by MC = V*NW
    V = [[[None for t in wDisc] for s in wTime] for d in range(len(C))]
    ZMC = [[None for t in wDisc] for s in wTime]
    MC = [[None for s in wTime] for d in range(len(C))]
    for d in range(len(C)):
        for s in range(len(wTime)):
            MC[d][s] = model.addVar(lb = 0.0, name = 'MC_'+str(d)+str(wTime[s]))
            for t in range(len(wDisc)):
                V[d][s][t] = model.addVar(lb = 0.0, name = 'V_'+str(d)+str(wTime[s])+'_'+str(wDisc[t]))
    for s in range(len(wTime)):          
        for t in range(len(wDisc)):
            ZMC[s][t] = model.addVar(vtype = GRB.BINARY, name = 'ZMC_'+str(d)+str(wTime[s])+'_'+str(wDisc[t]))
    return V, ZMC, MC, wDisc, wTime


#


# Discrete approximation constraints

def gen_WBCBinaryApprox(model, C, NW, MC, V, ZMC, wDisc, wTime, maxConcs, wbcDelay, eulerH):  
    conc_delay = int(wbcDelay/eulerH)
    numDaysForAvg = 1.0
    wAvgOver = int(numDaysForAvg/eulerH)
    for s in range(len(wTime)):
        for t in range(len(wDisc)):
            model.addConstr(NW[s] - sum(wDisc[t]*ZMC[s][t] for t in range(len(wDisc))) <= (wDisc[1] - wDisc[0])/2)
            model.addConstr(-NW[s] + sum(wDisc[t]*ZMC[s][t] for t in range(len(wDisc))) <= (wDisc[1] - wDisc[0])/2)
            model.addConstr(sum(ZMC[s][t] for t in range(len(wDisc))) == 1)
    for d in range(len(C)):
        for s in range(len(wTime)):
            model.addConstr(MC[d][s] == sum(wDisc[t]*V[d][s][t] for t in range(len(wDisc))))
            for t in range(len(wDisc)):
                model.addConstr(V[d][s][t] <= maxConcs[d]*ZMC[s][t])
                # To avoid drastic behavior of MC relaxations, the (delayed) drug concentration is an average over numDaysForAvg days (dafault value is 1 day.)
                model.addConstr(V[d][s][t] <= (1/wAvgOver)*(sum(C[d][s - conc_delay + l] for l in range(wAvgOver))))
                model.addConstr(V[d][s][t] >= (1/wAvgOver)*(sum(C[d][s - conc_delay + l] for l in range(wAvgOver))) + maxConcs[d]*(ZMC[s][t] - 1))


#


# WBC ODE discretization and approximation

def gen_WBCODEConstraints(model, NW, C, eulerH, wbcTurnover, wbcProduction, wbcDelay, wbcEffect, wbcInitial, wbcMin, wTimeDisc, maxConcs, wbcNeutrophil, wbcLymphocyte, wbcNeutrophilMin, wbcLymphocyteMin):
    #wbcDelay is in days, so the number of delay steps is wbcDelay/(eulerH*wTimeDisc)
    stepDelay = int(wbcDelay/(eulerH*wTimeDisc))
    V, ZMC, MC, wDisc, wTime = gen_WBCBinaryModelVars(model, C, eulerH, wbcMin, wbcInitial, wTimeDisc)
    gen_WBCBinaryApprox(model, C, NW, MC, V, ZMC, wDisc, wTime, maxConcs, wbcDelay, eulerH)
    # Up until stepDelay, just normal dynamics:
    for s in range(stepDelay):
        model.addConstr(NW[s+1] == NW[s] + wTimeDisc*eulerH*(wbcProduction - wbcTurnover*NW[s]))
        #model.addConstr(NW[s] >= wbcMin)
        model.addConstr(wbcNeutrophil*NW[s] >= wbcNeutrophilMin)
        model.addConstr(wbcLymphocyte*NW[s] >= wbcLymphocyteMin)
    # After stepDelay
    for s in range(stepDelay,len(NW) - 1):
        model.addConstr(NW[s+1] == NW[s] + wTimeDisc*eulerH*(wbcProduction - wbcTurnover*NW[s] - sum(wbcEffect[d]*MC[d][s] for d in range(len(C)))))
        #model.addConstr(NW[s] >= wbcMin)
        model.addConstr(wbcNeutrophil*NW[s] >= wbcNeutrophilMin)
        model.addConstr(wbcLymphocyte*NW[s] >= wbcLymphocyteMin)
    model.addConstr(wbcNeutrophil*NW[len(NW) - 1] >= wbcNeutrophilMin)
    model.addConstr(wbcLymphocyte*NW[len(NW) - 1] >= wbcLymphocyteMin)
    # WBC boundary constraint
    model.addConstr(NW[0] == wbcInitial)
    return MC


#


# WBC main subroutine

def gen_WBCConstraintsAndVars(model, C, eulerH, opParams):
    if 'wbcInfo' in opParams and 'maxConcs' in opParams:
        maxConcs = opParams['maxConcs']
        wbcInfo = opParams['wbcInfo']
        wbcInitial = wbcInfo[0][0]
        wbcTurnover = wbcInfo[0][1]
        wbcProduction = wbcInfo[0][2]
        wbcEffect = wbcInfo[1][0]
        wbcDelay = wbcInfo[1][1]
        wbcMin = wbcInfo[1][2]
        wbcNeutrophilMin = wbcInfo[1][3]
        wbcLymphocyteMin = wbcInfo[1][4]
        wTimeDisc = wbcInfo[1][5]
        wbcNeutrophil = wbcInfo[2][0]
        wbcLymphocyte = wbcInfo[2][1]
        NW = gen_WBCPopVars(model, C, wTimeDisc)
        MC = gen_WBCODEConstraints(model, NW, C, eulerH, wbcTurnover, wbcProduction, wbcDelay, wbcEffect, wbcInitial, wbcMin, wTimeDisc, maxConcs, wbcNeutrophil, wbcLymphocyte, wbcNeutrophilMin, wbcLymphocyteMin)
        return [NW, MC]


#


# Operational constraints (and variables) main subroutine

def gen_OperationalConstraints(model, U, C, E, eulerH, vol, daySets, mealSets, opParams, pillRegularize):
    newVars = {} 
    gen_MaxDoseConstraints(model, U, opParams)
    gen_MaxCumDailyDoseConstraints(model, U, opParams, daySets)  
    gen_InfusionRateConstraints(model, U, eulerH, opParams)  
    gen_MaxConcentrationConstraints(model, C, vol, opParams)  
    ZPill = gen_PillAdminVarsAndConstraints(model, U, opParams, mealSets, pillRegularize)  
    newVars['pill'] = ZPill
    ZRest = gen_RestDaysConstraintsAndVars(model, U, opParams, daySets)
    newVars['rest'] = ZRest
    ZEff = gen_RELUEffectiveVarsAndConstraints(model, C, E, opParams) 
    newVars['eff'] = ZEff
    WBCVars = gen_WBCConstraintsAndVars(model, C, eulerH, opParams)
    newVars['NW'] = WBCVars[0]
    newVars['MC'] = WBCVars[1]
    return newVars


# ### Plot

#


def allPlot(P, U, C, E, timeSteps, MC, NW, eulerH, wbcNeutrophil, wbcLymphocyte, wbcNeutrophilMin, wbcLymphocyteMin):
    NW = [NW[s].x for s in range(len(NW))]
    MC = [[MC[d][s].x for s in range(len(MC[d]))] for d in range(len(MC))]
    dNames = ['capecitabine', 'docetaxel', 'etoposide']
    wbcNames = ['Neutrophil', 'Clinical bound (Neutrophil)', 'Lymphocyte', 'Clinical bound (Lymphocyte)']
    popNames = ['tumor-N', 'tumor-C', 'tumor-D', 'tumor-E', 'tumor-Total']
    #
    markerPos = [s for s in range(len(timeSteps)) if s%(3.0/eulerH) == 0]
    markerPos.append(len(timeSteps)-1)
    myMarkers=['o','d','<','+']
    myLinestyles=[(0, (5, 5)), 'dashdot', 'dotted', (0, (3, 5, 1, 5, 1, 5))]
    #
    for e in range(len(P)): 
        for q in range(len(P[0])):
            temp = [math.exp(P[e][q][s].x) for s in range(len(P[0][0]))]
            plt.plot(timeSteps, temp, linestyle=myLinestyles[q], marker=myMarkers[q], markevery=markerPos, mfc='none') 
        ALL = [sum(math.exp(P[e][q][s].x) for q in range(len(P[0]))) for s in range(len(P[0][0]))]
        plt.plot(timeSteps, ALL, linestyle='solid', marker='s', markevery=markerPos)
        plt.xticks(range(1+int(max(timeSteps))))
        plt.legend(popNames)
        plt.title('Tumor Cell Population: Scenario '+str(e+1))
        plt.xlabel('Time (day)')
        plt.ylabel('Cell count')
        plt.savefig('Pop_scenario'+str(e+1)+'_alpha.png')
        plt.close()
    #
    for e in range(len(P)):
        for q in range(len(P[0])):
            temp = [P[e][q][s].x for s in range(len(P[0][0]))]
            plt.plot(timeSteps, temp, linestyle=myLinestyles[q], marker=myMarkers[q], markevery=markerPos, mfc='none') 
        ALL = [math.log(sum(math.exp(P[e][q][s].x) for q in range(len(P[0])))) for s in range(len(P[0][0]))]
        plt.plot(timeSteps, ALL, linestyle='solid', marker='s', markevery=markerPos)
        plt.xticks(range(1+int(max(timeSteps))))
        plt.legend(popNames)
        plt.title('Tumor Cell Population: Scenario '+str(e+1))
        plt.xlabel('Time (day)')
        plt.ylabel('Log(Cells)')
        plt.savefig('logPop_scenario'+str(e+1)+'_alpha.png')
        plt.close()
    #
    for d in range(len(U)):
        temp = [1000*U[d][s].x for s in range(len(U[0]))]
        plt.plot(timeSteps, temp)
    plt.xticks(range(1+int(max(timeSteps))))
    plt.legend(dNames)
    plt.title('Drug Administration')
    plt.xlabel('Time (day)')
    plt.ylabel(r'Drug Administration (mg)')
    plt.savefig("DrugAdmin_alpha.png")
    plt.close()
    #
    for d in range(len(C)):
        temp = [C[d][s].x for s in range(len(C[0]))]
        plt.plot(timeSteps, temp)
    plt.xticks(range(1+int(max(timeSteps))))
    plt.legend(dNames)
    plt.title('Drug Concentration')
    plt.xlabel('Time (day)')
    plt.ylabel(r'Concentration (gr/m$^3$)')
    plt.savefig("Conc_alpha.png")
    plt.close()
    #
    drug = ['Capecitabine', 'Docetaxel', 'Etoposide']
    for d in range(len(U)):
        temp = [1000*U[d][s].x for s in range(len(C[0]))]
        plt.plot(timeSteps, temp)   
        plt.xticks(range(1+int(max(timeSteps))))
        plt.title(drug[d])
        plt.xlabel('Time (day)')
        plt.ylabel(r'Drug Administration (mg)')
        plt.savefig("U_"+str(d)+"_alpha.png")
        plt.close()
    #
    for d in range(len(C)):
        temp = [1000*C[d][s].x for s in range(len(C[0]))]
        plt.plot(timeSteps, temp)
        temp = [1000*E[d][s].x for s in range(len(E[0]))]
        plt.plot(timeSteps, temp)    
        plt.xticks(range(1+int(max(timeSteps))))
        plt.title(drug[d])
        plt.xlabel('Time (day)')
        plt.ylabel(r'Concentration vs. Effective Concentration (mg)')
        plt.savefig("C_E_"+str(d)+"_alpha.png")
        plt.close()
    #
    wTimeDisc = len(timeSteps)/len(NW)
    wTime = [timeSteps[s] for s in range(len(timeSteps)) if s%wTimeDisc == 0]
    wTime.append(wTime[-1]+1)
    NW.append(NW[-1])
    plt.plot(wTime, [wbcNeutrophil*(ele*10**9) for ele in NW], linestyle='solid', marker='s', markevery=3, mfc='none')
    plt.plot(wTime, [wbcNeutrophilMin*10**9 for ele in NW], linestyle=(0, (5, 10)), linewidth=0.9)
    plt.plot(wTime, [wbcLymphocyte*(ele*10**9) for ele in NW], linestyle='solid', marker='o', markevery=3, mfc='none')
    plt.plot(wTime, [wbcLymphocyteMin*10**9 for ele in NW], linestyle=(0, (1, 10)), linewidth=0.9)
    plt.xticks(range(1+int(max(wTime))))
    plt.xlabel('Time (days)')
    plt.ylabel(r'White Blood Cells/$m^3$')
    plt.legend(wbcNames)
    plt.title('White Blood Cell Population')
    plt.savefig('WBC_alpha.png')
    plt.close()


# ### MAIN

#


def sto_Chemo(treatmentLength, eulerH, scenarios, popTypes, drugTypes, bioCons, vol, gompParam, popMaxes, initPops, scenarioProb, surgerySize, effectCoeffs, effectExpons, opParams, pillRegularize, _gap):      
    # 
    theModel = Model()
    theModel.Params.TIME_LIMIT = 2*3600.0
    # GAP for debugging purposes
    if _gap > 1*10**(-4):
        theModel.Params.MIPGap = _gap
    #
    timeSteps, stageSteps = gen_TimeSteps(treatmentLength, eulerH)
    daySets, mealSets = gen_daySetsAndMeals(treatmentLength, eulerH)
    alpha = theModel.addVar(lb = 0.0, ub = 1.0, name = "Alpha")
    P = sto_PopulationVars(theModel, scenarios, popTypes, stageSteps)
    U = gen_DrugAdminVars(theModel, drugTypes, stageSteps)
    C = gen_DrugConcenVars(theModel, drugTypes, stageSteps)
    E = gen_DrugEffectVars(theModel, drugTypes, stageSteps)
    ZScenario = sto_ScenarioVars(theModel, scenarios)
    #
    theModel.setObjective(alpha, GRB.MAXIMIZE)
    #
    gen_ConcenODEConstraints(theModel, C, U, eulerH, bioCons, vol)
    sto_PopODEConstraints(theModel, P, E, eulerH, gompParam, popMaxes, effectCoeffs, effectExpons, initPops, timeSteps) 
    sto_chanceConstraints(theModel, alpha, P, ZScenario, popMaxes, initPops, surgerySize, scenarioProb)
    newVars = gen_OperationalConstraints(theModel, U, C, E, eulerH, vol, daySets, mealSets, opParams, pillRegularize)
    MC = [[0 for s in range(len(P[0]))] for d in range(len(U))]
    NW = [0 for ele in C[0]]
    if 'NW' in newVars and 'MC' in newVars:
        NW = newVars['NW']; MC = newVars['MC']
    else:
        print('*** NewVars of Operational Constraints were NOT generated! *** \n \n')
        input('Press any key to continue...')
    # SOLVE
    theModel.write('sto_alpha_Chemo_formulation.lp')
    status = theModel.optimize()
    obj = theModel.getObjective()
    print("\n \n************************************")
    print("** Obj value = ",obj.getValue(),"**")
    print("************************************\n \n")
    theModel.write("sto_alpha_Chemo_solution.sol")
    # PLOT
    if 'wbcInfo' in opParams:
        wbcInfo = opParams['wbcInfo']
        wbcNeutrophilMin = wbcInfo[1][3]
        wbcLymphocyteMin = wbcInfo[1][4]
        wbcNeutrophil = wbcInfo[2][0]
        wbcLymphocyte = wbcInfo[2][1]
        allPlot(P, U, C, E, timeSteps, MC, NW, eulerH, wbcNeutrophil, wbcLymphocyte, wbcNeutrophilMin, wbcLymphocyteMin) 
    # Output
    stageSteps_out = stageSteps
    timeSteps_out = timeSteps
    daySets_out = daySets
    mealSets_out = mealSets
    P_out = [[[P[e][q][s].x for s in stageSteps] for q in popTypes] for e in scenarios]
    U_out = [[1000*U[d][s].x for s in stageSteps] for d in drugTypes]         # in mg
    C_out = [[1000*C[d][s].x for s in stageSteps] for d in drugTypes]         # in mg
    E_out = [[1000*E[d][s].x for s in stageSteps] for d in drugTypes]         # in mg
    wTimeDisc = len(timeSteps)/len(NW)
    wTime_out = [timeSteps[s] for s in range(len(timeSteps)) if s%wTimeDisc == 0]
    wTime_out.append(wTime_out[-1]+1)
    NW_out = [NW[s].x for s in range(len(NW))]
    NW_out.append(NW_out[-1]) 
    wbcNeutrophil_out = [wbcNeutrophil*(ele*10**9) for ele in NW_out]
    wbcLymphocyte_out = [wbcLymphocyte*(ele*10**9) for ele in NW_out]
    wbcNeutrophilMin_out = [wbcNeutrophilMin*10**9 for ele in NW_out]
    wbcLymphocyteMin_out = [wbcLymphocyteMin*10**9 for ele in NW_out]
    output = [stageSteps_out, timeSteps_out, daySets_out, mealSets_out, P_out, U_out, C_out, E_out, wTime_out, NW_out, wbcNeutrophil_out, wbcLymphocyte_out, wbcNeutrophilMin_out, wbcLymphocyteMin_out]
    return output


#


def run_sto_Chemo(**kwargs):                      # may input GAP for debugging purposes, e.g., gap=0.005 for 0.5% gap -- no input equals no gap
    #
    treatmentLength = 21                          # in days
    eulerH = 1.0/(8*3.0)                          # as a fraction of a day (e.g., 1/24 for one hour)
    drugTypes = range(0,3)
    popTypes = range(0,4)
    scenarios = range(0,10)
    #
    bioCons = [.6,.2,.8]
    vol = 15*10**-3                               # volume (in m^3) for drug concentration
    gompParam = 0.0007
    initPops = [[20.53, 17.89, 17.89, 17.89],
                [20.44, 17.85, 17.83, 18.69],
                [20.44, 18.70, 17.86, 17.86],
                [20.44, 17.84, 18.74, 17.82],
                [20.22, 19.50, 17.74, 17.72],
                [20.23, 17.71, 19.49, 17.71],
                [20.25, 17.72, 17.67, 19.46],
                [19.80, 17.35, 17.45, 20.09],
                [19.81, 17.20, 20.10, 17.28],
                [19.80, 20.11, 17.18, 17.39]]
    popMaxes = [[7.0 + ele for ele in scenario] for scenario in initPops]
    scenarioProb = [0.7705, 0.0619, 0.0603, 0.0579, 0.0109, 0.0109, 0.0103, 0.0064, 0.0059, 0.0050]
    surgerySize = 19.81
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
    maxConcs = [7.10, 0.17, 0.12] 
    maxConcs = [ele/vol for ele in maxConcs]
    patientVolume = 1.7
    doseCaps = [1.26, 10.0, 0.03]
    doseCaps = [ele*patientVolume for ele in doseCaps]
    dailyDoseCaps = [2.51, 0.10, 0.06]
    dailyDoseCaps = [ele*patientVolume for ele in dailyDoseCaps]
    infusionRates = [10.0, 0.1, 10.0] 
    infusionRates = [ele*patientVolume for ele in infusionRates]
    pillSizes = [.50 , .00, .05]
    effectFloors = [.0, .0, .5]
    useEffectiveConcs = [2]
    restDays = [0, 7, 0]
    ## WBC 
    wbcInitial = 8.0*10**3                        # For numerical reasons the unit is (billion cell per cubic meter)
    wbcTurnover = 0.15
    wbcProduction = 1.2*10**3                     # For numerical reasons the unit is (billion cell per cubic meter)
    wbcGrowth = [wbcInitial, wbcTurnover, wbcProduction]
    wbcEffect = [cEffect, dEffect, eEffect]       # conservative assumption: kill effect on WBC equal to tumor
    wbcDelay = 5  
    wbcMin = 3.0*10**3                            # For numerical reasons the unit is (billion cell per cubic meter)
    wbcNeutrophilMin = 2.5*10**3                  # For numerical reasons the unit is (billion cell per cubic meter)
    wbcLymphocyteMin = 1.0*10**3                  # For numerical reasons the unit is (billion cell per cubic meter)
    wbcTimeDisc = int(1.0/eulerH)
    wbcDrugs = [wbcEffect, wbcDelay, wbcMin, wbcNeutrophilMin, wbcLymphocyteMin, wbcTimeDisc]
    wbcNeutrophil = 0.5
    wbcLymphocyte = 0.3
    wbcCompos = [wbcNeutrophil, wbcLymphocyte]
    wbcInfo = [wbcGrowth, wbcDrugs, wbcCompos]
    #
    opParams = {'doseCaps': doseCaps, 'dailyDoseCaps': dailyDoseCaps, 'infusionRates': infusionRates, 'pillSizes': pillSizes, \
                'effectFloors': effectFloors, 'maxConcs': maxConcs, 'useEffectiveConcs': useEffectiveConcs, \
                'restDays': restDays, 'wbcInfo': wbcInfo}
    #
    pillRegularize = 'no'
    # MAIN ROUTINE
    if 'gap' in kwargs:
        _gap = kwargs['gap']
    else:
        _gap = 0.0
    output = sto_Chemo(treatmentLength, eulerH, scenarios, popTypes, drugTypes, bioCons, vol, gompParam, popMaxes, initPops, scenarioProb, surgerySize, effectCoeffs, effectExpons, opParams, pillRegularize, _gap)
    output.append(eulerH)
    output.append(vol)
    return output


#


output = run_sto_Chemo()

