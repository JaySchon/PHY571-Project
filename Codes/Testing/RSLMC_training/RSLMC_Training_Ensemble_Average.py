# This file is aimed to train SLMC model using Ensemble Average.
# First order eff_param calculated from Local Update Algorithm are used to create more samples at T = Tc, and train more eff_param iteratively;
# Finally, we take the average of all the params to get ensemble average.


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
from SLMC_Training_Lib import restricted_self_optimization
import time
import sys
sys.setrecursionlimit(3000)

size  = [25, 40, 60, 80]
J = 1.
K = 0.2
T = 2.5
Nsamples = 2500
warmup = 1000
interval = [12, 20, 30, 40] # To be modified

for i in range(len(size)):
    print('----------------- Size = %i -------------------'%size[i])
    eff_param = np.loadtxt('Local_data_fitting_eff_param(n=1)(L=%i).dat'%size[i])
    time1 = time.time()
    # Set iteration step
    Iter = 30 # to be modified
    # Calculate eff_param for "Iter" time
    restricted_self_optimization(Iter, size[i], J, K, T, Nsamples, warmup, interval[i], eff_param)
    time2 = time.time()
    print('################# Size = %i ###################'%size[i])
    print('Size = %i done, time = %i'%(size[i], time2-time1))


