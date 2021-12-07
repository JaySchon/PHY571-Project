# This file is to calculate the flipping probability when constructing the cluster in SLMC and RSLMC method.

################################ Output Reminder #######################################
#                                                          #  
#      Calculated result will be printed and displayed in your terminal;         #
#          You must save those output result by yourself !                 #
#               Otherwise you will lose them !                       #
#         In Linux system, you can use "tee" command to save them;            #
# For example, run with ">> python SLMC_Training_Ensemble_Average.py |tee result.txt"  #
# Correspondingly, you can find solutions in other systems to save printed results.   #
#     Note: Data dealing script is not uploaded, you can write it by yourself.     #
########################################################################################

#import libraries
import numpy as np
import numpy.random as rnd
from Configuration import Configuration
from SelfLearningUpdate_modified import SelfLearningUpdate
from RestrictedSelfLearningUpdate_modified import RestrictedSelfLearningUpdate
import time
import sys
sys.setrecursionlimit(3000)

# Calculate flipping prob of SLMC method.
size = [25, 40, 60]
J = 1
K = 0.2
T = 2.5
# eff_param obtained from Temp descent method.
eff_param = [[18.20677823,  1.10828245], [43.85759863,  1.10707017], [96.84205157,  1.1069041 ]]
print('-------------SLMC Flipping Probability----------------')
for i in range(len(size)):
    L = size[i]
    t1 = time.time()
    n_cycles = 20000
    spins = rnd.choice([-1,1],size = (L, L))
    config = Configuration(spins, L, J, K, T)
    count = 0
    # Monte Carlo
    for n in range(n_cycles):
        update = SelfLearningUpdate(spins, J, K, T, eff_param[i])
        spins = update.SLMC_Update()[0]
        if update.SLMC_Update()[1] == True:
            count += 1  
    acceptance = count / n_cycles
    print('Flipping Probability of SLMC(L=%i):'%L, acceptance)
    t2 = time.time()
    print('Size = %i, time = %.4f.'%(L,t2-t1))

# Calculate flipping prob of RSLMC method.

size = [25, 40, 60, 80, 100, 120]
J = 1
K = 0.2
T = 2.5
restriction = [10, 15, 25, 35, 40, 40]
# eff_param obtained from Temp descent method.
eff_param = [[17.3016049,   1.10735113], [42.71708316,  1.1067934 ], [96.16359073,  1.1068075 ],\
         [112.58510281, 1.10005775], [153.45171008,  1.09884171], [228.38157127, 1.09935068] ]

print('-------------RSLMC Flipping Probability----------------')
for i in range(len(size)):
    L = size[i]
    t1 = time.time()
    n_cycles = 20000
    count = 0
    spins = rnd.choice([-1,1],size = (L, L))

    # Monte Carlo
    for n in range(n_cycles):
        update = RestrictedSelfLearningUpdate(spins, J, K, T, eff_param[i], restriction[i])
        spins = update.Restricted_SLMC_Update()[0]
        if update.Restricted_SLMC_Update()[1] == True:
            count += 1
    acceptance = count / n_cycles
    print('Flipping Prob of RSLMC(L=%i):'%L, acceptance)
    t2 = time.time()
    print('Size = %i, time = %.4f.'%(L,t2-t1))

    