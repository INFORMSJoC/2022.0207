#  ___________________________________________________________________________________________________________
# |                                                                                                           |
# |                       Combination Chemotherapy Optimization with Discrete Dosing                          |
# |                                                                                                           |
# |           Temitayo Ajayi, Seyedmohammadhossein Hosseinian, Andrew J. Schaefer, Clifton D. Fuller          |
# |___________________________________________________________________________________________________________|


# # Sensitivity analysis

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

def det_TimeSteps(treatmentLength, eulerH):
    timeSteps = np.linspace(0,treatmentLength,math.ceil(treatmentLength/eulerH))
    stageSteps = []
    counter = 0
    for s in timeSteps:
        stageSteps.append(counter)
        counter += 1
    return timeSteps, stageSteps


#


# Partitioning the "stageSteps" into days (outputs "daySets") as well as indices for meals (outputs "mealSets")

def det_daySetsAndMeals(treatmentLength, eulerH):
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

def det_PopulationVars(model, popTypes, stageSteps):
    P = [[None for s in stageSteps] for q in popTypes]
    for q in popTypes:
        for s in stageSteps:
            P[q][s] = model.addVar(lb = -float("inf"), ub = float("inf"), name = "P_"+str(q)+"_"+str(s))
    return P


#


# Drug administration variables

def det_DrugAdminVars(model, drugTypes, stageSteps): 
    U = [[None for s in stageSteps] for d in drugTypes]
    for d in drugTypes:
        for s in stageSteps:
            U[d][s] = model.addVar(lb = 0.0, ub = float("inf"), name = "U_"+str(d)+"_"+str(s))
    return U


#


# Drug concentration variables

def det_DrugConcenVars(model, drugTypes, stageSteps):
    C = [[None for s in stageSteps] for d in drugTypes]
    for d in drugTypes:
        for s in stageSteps:
            C[d][s] = model.addVar(lb = 0.0, ub = float("inf"), name = "C_"+str(d)+"_"+str(s))
    return C


#


# Drug effective-concentration variables (effective concentration of the drugs depends on the cell types)

def det_DrugEffectVars(model, drugTypes, stageSteps):
    E = [[None for s in stageSteps] for d in drugTypes]
    for d in drugTypes:
        for s in stageSteps:
            E[d][s] = model.addVar(lb = 0.0, ub = float("inf"), name = "E_"+str(d)+"_"+str(s))
    return E


#


# ODE discretization: drug concentration

def det_ConcenODEConstraints(model, C, U, eulerH, bioCons, vol):
    drugTypes = range(len(U))
    stageStepsLessOne = range(len(U[0]) - 1)
    for d in drugTypes:
        model.addConstr(C[d][0] == 0)
        for s in stageStepsLessOne:
            model.addConstr(C[d][s+1] == C[d][s] - (eulerH * bioCons[d] * C[d][s]) + (U[d][s] / vol))


#


# ODE discretization: tumor dynamics

def det_PopODEConstraints(model, P, E, eulerH, gompParam, popMaxes, effectCoeffs, effectExpons, initPops, timeSteps):
    popTypes = range(len(P))
    drugTypes = range(len(E))
    stageStepsLessOne = range(len(E[0]) - 1)
    for q in popTypes:
        model.addConstr(P[q][0] == initPops[q])
        for s in stageStepsLessOne:
            model.addConstr(P[q][s + 1] == P[q][s] + eulerH*(gompParam*(popMaxes[q] - P[q][s])\
            - sum(effectCoeffs[d][q]*math.exp(-effectExpons[d][q]*timeSteps[s])*E[d][s] for d in drugTypes)))


# ### Operational constraints (and variables)

#


# Max dose per time step

def det_MaxDoseConstraints(model, U, opParams):
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

def det_MaxCumDailyDoseConstraints(model, U, opParams, daySets):
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

def det_InfusionRateConstraints(model, U, eulerH, opParams):
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

def det_MaxConcentrationConstraints(model, C, vol, opParams):
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

def det_PillAdminVarsAndConstraints(model, U, opParams, mealSets, pillRegularize):
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

def det_RestDaysConstraintsAndVars(model, U, opParams, daySets):
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

def det_RELUEffectiveVarsAndConstraints(model, C, E, opParams):
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

def det_WBCPopVars(model, C, wTimeDisc):
    wTime = [s for s in range(len(C[0])) if s%wTimeDisc == 0]
    NW = [None for s in wTime]
    for s in range(len(wTime)):
        NW[s] = model.addVar(lb = 0.0, ub = float("inf"), name = 'NW_'+str(s))
    return NW


#


# Binary and continuous variables for discrete approximation

def det_WBCBinaryModelVars(model, C, eulerH, wbcMin, wbcInitial, wTimeDisc):
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

def det_WBCBinaryApprox(model, C, NW, MC, V, ZMC, wDisc, wTime, maxConcs, wbcDelay, eulerH):  
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

def det_WBCODEConstraints(model, NW, C, eulerH, wbcTurnover, wbcProduction, wbcDelay, wbcEffect, wbcInitial, wbcMin, wTimeDisc, maxConcs, wbcNeutrophil, wbcLymphocyte, wbcNeutrophilMin, wbcLymphocyteMin):
    #wbcDelay is in days, so the number of delay steps is wbcDelay/(eulerH*wTimeDisc)
    stepDelay = int(wbcDelay/(eulerH*wTimeDisc))
    V, ZMC, MC, wDisc, wTime = det_WBCBinaryModelVars(model, C, eulerH, wbcMin, wbcInitial, wTimeDisc)
    det_WBCBinaryApprox(model, C, NW, MC, V, ZMC, wDisc, wTime, maxConcs, wbcDelay, eulerH)
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

def det_WBCConstraintsAndVars(model, C, eulerH, opParams):
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
        NW = det_WBCPopVars(model, C, wTimeDisc)
        MC = det_WBCODEConstraints(model, NW, C, eulerH, wbcTurnover, wbcProduction, wbcDelay, wbcEffect, wbcInitial, wbcMin, wTimeDisc, maxConcs, wbcNeutrophil, wbcLymphocyte, wbcNeutrophilMin, wbcLymphocyteMin)
        return [NW, MC]


#


# Operational constraints (and variables) main subroutine

def det_OperationalConstraints(model, U, C, E, eulerH, vol, daySets, mealSets, opParams, pillRegularize):
    newVars = {} 
    det_MaxDoseConstraints(model, U, opParams) 
    det_MaxCumDailyDoseConstraints(model, U, opParams, daySets)  
    det_InfusionRateConstraints(model, U, eulerH, opParams)  
    det_MaxConcentrationConstraints(model, C, vol, opParams)  
    ZPill = det_PillAdminVarsAndConstraints(model, U, opParams, mealSets, pillRegularize)  
    newVars['pill'] = ZPill
    ZRest = det_RestDaysConstraintsAndVars(model, U, opParams, daySets)
    newVars['rest'] = ZRest
    ZEff = det_RELUEffectiveVarsAndConstraints(model, C, E, opParams) 
    newVars['eff'] = ZEff
    WBCVars = det_WBCConstraintsAndVars(model, C, eulerH, opParams)
    newVars['NW'] = WBCVars[0]
    newVars['MC'] = WBCVars[1]
    return newVars


# ### MAIN

#


def det_Chemo(priority, treatmentLength, eulerH, popTypes, drugTypes, bioCons, vol, gompParam, popMaxes, initPops, effectCoeffs, effectExpons, opParams, pillRegularize, _gap): 
    #
    theModel = Model()
    theModel.Params.TIME_LIMIT = 1.25*3600.0
    #theModel.Params.LogToConsole = 0
    # GAP for debugging purposes
    if _gap > 1*10**(-4):
        theModel.Params.MIPGap = _gap
    #
    timeSteps, stageSteps = det_TimeSteps(treatmentLength, eulerH)
    daySets, mealSets = det_daySetsAndMeals(treatmentLength, eulerH)
    P = det_PopulationVars(theModel, popTypes, stageSteps)
    U = det_DrugAdminVars(theModel, drugTypes, stageSteps)
    C = det_DrugConcenVars(theModel, drugTypes, stageSteps)
    E = det_DrugEffectVars(theModel, drugTypes, stageSteps)
    #
    theModel.setObjective(sum(priority[q]*P[q][len(stageSteps) - 1] for q in popTypes), GRB.MINIMIZE)
    #
    det_ConcenODEConstraints(theModel, C, U, eulerH, bioCons, vol)
    det_PopODEConstraints(theModel, P, E, eulerH, gompParam, popMaxes, effectCoeffs, effectExpons, initPops, timeSteps)   
    newVars = det_OperationalConstraints(theModel, U, C, E, eulerH, vol, daySets, mealSets, opParams, pillRegularize)
    MC = [[0 for s in range(len(P[0]))] for d in range(len(U))]
    NW = [0 for ele in C[0]]
    if 'NW' in newVars and 'MC' in newVars:
        NW = newVars['NW']; MC = newVars['MC']
    else:
        print('*** NewVars of Operational Constraints were NOT generated! *** \n \n')
        input('Press any key to continue...')
    # SOLVE
    status = theModel.optimize()
    obj = theModel.getObjective()
    print("\n \n************************************")
    print("** Obj value = ",obj.getValue(),"**")
    print("************************************\n \n")
    #
    if 'wbcInfo' in opParams:
        wbcInfo = opParams['wbcInfo']
        wbcNeutrophilMin = wbcInfo[1][3]
        wbcLymphocyteMin = wbcInfo[1][4]
        wbcNeutrophil = wbcInfo[2][0]
        wbcLymphocyte = wbcInfo[2][1] 
    # Output
    stageSteps_out = stageSteps
    timeSteps_out = timeSteps
    daySets_out = daySets
    mealSets_out = mealSets
    P_out = [[P[q][s].x for s in stageSteps] for q in popTypes]
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
    return output, obj.getValue()


#


def run_sen_Chemo(**kwargs):                      # may input GAP for debugging purposes, e.g., gap=0.005 for 0.5% gap -- no input equals no gap
    #
    priority = [1,1,1,1]
    treatmentLength = 21                          # in days
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
    opParams = {'doseCaps': doseCaps,'dailyDoseCaps': dailyDoseCaps, 'infusionRates': infusionRates, 'pillSizes': pillSizes, \
                'effectFloors': effectFloors, 'maxConcs': maxConcs, 'useEffectiveConcs': useEffectiveConcs, \
                'restDays': restDays, 'wbcInfo': wbcInfo}
    #
    pillRegularize = 'no'
    # MAIN ROUTINE
    if 'gap' in kwargs:
        _gap = kwargs['gap']
    else:
        _gap = 0.0
    ALLoutput = [[] for d in drugTypes]
    ALLobj =[[] for d in drugTypes]
    if 'sen_killEffect' in kwargs:
        sen_killEffect = kwargs['sen_killEffect']
        for d in drugTypes:
            for sensitivityCoeff in sen_killEffect:
                temp_effectCoeffs = [[] for ele in effectCoeffs]
                for ele in range(len(effectCoeffs)):
                    temp_effectCoeffs[ele] = effectCoeffs[ele].copy()
                for ele in range(len(temp_effectCoeffs[d])):
                    temp_effectCoeffs[d][ele] *= sensitivityCoeff
                print('#####  Kill Effect: d_'+str(d)+'_senCoeff_'+str(sensitivityCoeff)+'  #####\n')
                output, obj = det_Chemo(priority, treatmentLength, eulerH, popTypes, drugTypes, bioCons, vol, gompParam, popMaxes, initPops, temp_effectCoeffs, effectExpons, opParams, pillRegularize, _gap)
                output.append(eulerH)
                output.append(vol)
                ALLoutput[d].append(output)   
                ALLobj[d].append(obj)
    if 'sen_temporalResist' in kwargs:
        sen_temporalResist = kwargs['sen_temporalResist']
        for d in drugTypes:
            for sensitivityCoeff in sen_temporalResist:
                temp_effectExpons = [[] for ele in effectExpons]
                for ele in range(len(effectExpons)):
                    temp_effectExpons[ele] = effectExpons[ele].copy()
                for ele in range(len(temp_effectExpons[d])):
                    temp_effectExpons[d][ele] *= sensitivityCoeff
                print('#####  Temporal Resistance: d_'+str(d)+'_senCoeff_'+str(sensitivityCoeff)+'  #####\n')
                output, obj = det_Chemo(priority, treatmentLength, eulerH, popTypes, drugTypes, bioCons, vol, gompParam, popMaxes, initPops, effectCoeffs, temp_effectExpons, opParams, pillRegularize, _gap)
                output.append(eulerH)
                output.append(vol)
                ALLoutput[d].append(output)   
                ALLobj[d].append(obj)
    if 'sen_bioConstant' in kwargs:
        sen_bioConstant = kwargs['sen_bioConstant']
        for d in drugTypes:
            for sensitivityCoeff in sen_bioConstant:
                temp_bioCons = bioCons.copy()
                temp_bioCons[d] *= sensitivityCoeff
                print('#####  Biological Constant: d_'+str(d)+'_senCoeff_'+str(sensitivityCoeff)+'  #####\n')
                output, obj = det_Chemo(priority, treatmentLength, eulerH, popTypes, drugTypes, temp_bioCons, vol, gompParam, popMaxes, initPops, effectCoeffs, effectExpons, opParams, pillRegularize, _gap)
                output.append(eulerH)
                output.append(vol)
                ALLoutput[d].append(output)   
                ALLobj[d].append(obj)
    if 'sen_wbcKill' in kwargs:
        sen_wbcKill = kwargs['sen_wbcKill']
        for d in drugTypes:
            for sensitivityCoeff in sen_wbcKill:   
                temp_wbcEffect = wbcEffect.copy()
                temp_wbcEffect[d] *= sensitivityCoeff
                temp_wbcGrowth = [wbcInitial, wbcTurnover, wbcProduction]
                temp_wbcDrugs = [temp_wbcEffect, wbcDelay, wbcMin, wbcNeutrophilMin, wbcLymphocyteMin, wbcTimeDisc]
                temp_wbcCompos = [wbcNeutrophil, wbcLymphocyte]
                temp_wbcInfo = [temp_wbcGrowth, temp_wbcDrugs, temp_wbcCompos]
                temp_opParams = {'doseCaps': doseCaps,'dailyDoseCaps': dailyDoseCaps, 'infusionRates': infusionRates, 'pillSizes': pillSizes, \
                                 'effectFloors': effectFloors, 'maxConcs': maxConcs, 'useEffectiveConcs': useEffectiveConcs, \
                                 'restDays': restDays, 'wbcInfo': temp_wbcInfo}      
                print('#####  WBC Effect: d_'+str(d)+'_senCoeff_'+str(sensitivityCoeff)+'  #####\n')
                output, obj = det_Chemo(priority, treatmentLength, eulerH, popTypes, drugTypes, bioCons, vol, gompParam, popMaxes, initPops, effectCoeffs, effectExpons, temp_opParams, pillRegularize, _gap)
                output.append(eulerH)
                output.append(vol)
                ALLoutput[d].append(output)   
                ALLobj[d].append(obj)
    if 'sen_maxAdmin' in kwargs:
        sen_maxAdmin = kwargs['sen_maxAdmin']
        for d in range(len(sen_maxAdmin)):
            for case in range(len(sen_maxAdmin[d])): 
                if d != 1:
                    temp_doseCaps = doseCaps.copy()
                    temp_doseCaps[d] = patientVolume*sen_maxAdmin[d][case][0]
                    temp_dailyDoseCaps = dailyDoseCaps.copy()
                    temp_dailyDoseCaps[d] = patientVolume*sen_maxAdmin[d][case][1]
                    temp_maxConcs = maxConcs.copy()
                    temp_maxConcs[d] = sen_maxAdmin[d][case][2]/vol
                else:
                    temp_doseCaps = doseCaps.copy()
                    temp_dailyDoseCaps = dailyDoseCaps.copy()
                    temp_dailyDoseCaps[d] = patientVolume*sen_maxAdmin[d][case][0]
                    temp_maxConcs = maxConcs.copy()
                    temp_maxConcs[d] = sen_maxAdmin[d][case][1]/vol
                temp_opParams = {'doseCaps': temp_doseCaps,'dailyDoseCaps': temp_dailyDoseCaps, 'infusionRates': infusionRates, 'pillSizes': pillSizes, \
                                 'effectFloors': effectFloors, 'maxConcs': temp_maxConcs, 'useEffectiveConcs': useEffectiveConcs, \
                                 'restDays': restDays, 'wbcInfo': wbcInfo}      
                print('#####  Max Daily Bound: d_'+str(d)+'_bound_'+str(round(temp_dailyDoseCaps[d]/patientVolume,2))+'_gr/m2  #####\n')
                output, obj = det_Chemo(priority, treatmentLength, eulerH, popTypes, drugTypes, bioCons, vol, gompParam, popMaxes, initPops, effectCoeffs, effectExpons, temp_opParams, pillRegularize, _gap)
                output.append(eulerH)
                output.append(vol)
                ALLoutput[d].append(output)   
                ALLobj[d].append(obj)
    if 'sen_NeutrophilFloor' in kwargs:
        sen_NeutrophilFloor = kwargs['sen_NeutrophilFloor']
        for sensitivityCoeff in sen_NeutrophilFloor:   
            temp_wbcNeutrophilMin = sensitivityCoeff*wbcNeutrophilMin
            temp_wbcGrowth = [wbcInitial, wbcTurnover, wbcProduction]
            temp_wbcDrugs = [wbcEffect, wbcDelay, wbcMin, temp_wbcNeutrophilMin, wbcLymphocyteMin, wbcTimeDisc]
            temp_wbcCompos = [wbcNeutrophil, wbcLymphocyte]
            temp_wbcInfo = [temp_wbcGrowth, temp_wbcDrugs, temp_wbcCompos]
            temp_opParams = {'doseCaps': doseCaps,'dailyDoseCaps': dailyDoseCaps, 'infusionRates': infusionRates, 'pillSizes': pillSizes, \
                                'effectFloors': effectFloors, 'maxConcs': maxConcs, 'useEffectiveConcs': useEffectiveConcs, \
                                'restDays': restDays, 'wbcInfo': temp_wbcInfo}      
            print('#####  Neutrophil Floor: senCoeff_'+str(sensitivityCoeff)+'  #####\n')
            output, obj = det_Chemo(priority, treatmentLength, eulerH, popTypes, drugTypes, bioCons, vol, gompParam, popMaxes, initPops, effectCoeffs, effectExpons, temp_opParams, pillRegularize, _gap)
            output.append(eulerH)
            output.append(vol)
            ALLoutput[0].append(output)   
            ALLobj[0].append(obj)                
    return ALLoutput, ALLobj


#


#******************************************************************************************************
# Please select "sensitivityType" from the following set:
# {sen_killEffect, sen_temporalResist, sen_bioConstant, sen_wbcKill, sen_maxAdmin, sen_NeutrophilFloor}
#******************************************************************************************************
sensitivityType = 'sen_NeutrophilFloor'
#
sensitivityCeoffs = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]
sensitivityAdminValue = [[[1.77, 3.54, 10.04], [1.48, 2.96, 8.40], [0.89, 1.78, 5.06], [0.59, 1.18, 3.35]],
                         [[0.150, 0.26], [0.125, 0.21], [0.075, 0.13], [0.050, 0.06]],
                         [[0.06, 0.09, 0.19], [0.03, 0.03, 0.06]]]
sensitivityAdminNames = [['3.54', '2.96', '1.78', '1.18'],
                         ['0.150', '0.125', '0.075', '0.050'],
                         ['0.09', '0.03']]                         
#
if sensitivityType == 'sen_killEffect':
    ALLoutput, ALLobj = run_sen_Chemo(sen_killEffect=sensitivityCeoffs)
    drugTypes = range(len(ALLoutput))
    senCases = range(len(ALLoutput[0]))
    print("\n \n************************************")
    print("\n         SENSITIVITY ANALYSIS      \n")
    print("\n          << Kill Effects >>      ")
    print("\n************************************")
    for d in drugTypes:
        for case in senCases:
            print('d_'+str(d)+'_senCoeff_'+str(sensitivityCeoffs[case])+': Objective value = ', ALLobj[d][case])
        print("----------------------------------------------------")
elif sensitivityType == 'sen_temporalResist':
    ALLoutput, ALLobj = run_sen_Chemo(sen_temporalResist=sensitivityCeoffs)
    drugTypes = range(len(ALLoutput))
    senCases = range(len(ALLoutput[0]))
    print("\n \n************************************")
    print("\n         SENSITIVITY ANALYSIS      \n")
    print("\n       << Temporal Resistance >>      ")
    print("\n************************************")
    for d in drugTypes:
        for case in senCases:
            print('d_'+str(d)+'_senCoeff_'+str(sensitivityCeoffs[case])+': Objective value = ', ALLobj[d][case])
        print("----------------------------------------------------")
elif sensitivityType == 'sen_bioConstant':
    ALLoutput, ALLobj = run_sen_Chemo(sen_bioConstant=sensitivityCeoffs)
    drugTypes = range(len(ALLoutput))
    senCases = range(len(ALLoutput[0]))
    print("\n \n************************************")
    print("\n         SENSITIVITY ANALYSIS      \n")
    print("\n       << Biological Constant >>      ")
    print("\n************************************")
    for d in drugTypes:
        for case in senCases:
            print('d_'+str(d)+'_senCoeff_'+str(sensitivityCeoffs[case])+': Objective value = ', ALLobj[d][case])
        print("----------------------------------------------------")
elif sensitivityType == 'sen_wbcKill':
    ALLoutput, ALLobj = run_sen_Chemo(sen_wbcKill=sensitivityCeoffs)
    drugTypes = range(len(ALLoutput))
    senCases = range(len(ALLoutput[0]))
    print("\n \n************************************")
    print("\n         SENSITIVITY ANALYSIS      \n")
    print("\n           << WBC Effect >>          ")
    print("\n************************************")
    for d in drugTypes:
        for case in senCases:
            print('d_'+str(d)+'_senCoeff_'+str(sensitivityCeoffs[case])+': Objective value = ', ALLobj[d][case])
        print("----------------------------------------------------")
elif sensitivityType == 'sen_maxAdmin':
    ALLoutput, ALLobj = run_sen_Chemo(sen_maxAdmin=sensitivityAdminValue)
    print("\n \n************************************")
    print("\n         SENSITIVITY ANALYSIS      \n")
    print("\n          << Daily Bound >>          ")
    print("\n************************************")
    for d in range(len(sensitivityAdminValue)):
        for case in range(len(sensitivityAdminValue[d])):
            print('d_'+str(d)+'_bound_'+str(sensitivityAdminNames[d][case])+'_grm-2: Objective value = ', ALLobj[d][case])
        print("----------------------------------------------------")
elif sensitivityType == 'sen_NeutrophilFloor':
    ALLoutput, ALLobj = run_sen_Chemo(sen_NeutrophilFloor=sensitivityCeoffs)
    senCases = range(len(ALLoutput[0]))
    print("\n \n************************************")
    print("\n         SENSITIVITY ANALYSIS      \n")
    print("\n        << Neutrophil Floor >>       ")
    print("\n************************************")
    for case in senCases:
        print('senCoeff_'+str(sensitivityCeoffs[case])+': Objective value = ', ALLobj[0][case])
    print("----------------------------------------------------")

