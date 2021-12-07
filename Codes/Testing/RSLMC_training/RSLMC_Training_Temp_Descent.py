# This file is aimed to train RSLMC model using Temperature descent method.
# Temperature lists are created, RSLMC training will be done at higher temperature first;
# Eff_param trained at high temp will be used as input, and train new eff_param at lower temperature;
# Finally, we will get eff_param at T = Tc.

################################ Output Reminder #######################################
#                                                                                      #  
#      Calculated result will be printed and displayed in your terminal;               #
#          You must save those output result by yourself !                             #
#               Otherwise you will lose them !                                         #
#         In Linux system, you can use "tee" command to save them;                     #
# For example, run with ">> python SLMC_Training_Ensemble_Average.py |tee result.txt"  #
# Correspondingly, you can find solutions in other systems to save printed results.    #
#     Note: Data dealing script is not uploaded, you can write it by yourself.         #
########################################################################################

#import libraries
import numpy as np
from SLMC_Training_Lib import Restricted_Temp_descend_opt
import time
import sys
sys.setrecursionlimit(3000)


size  = [25, 40, 60, 80, 100, 120, 150]
J = 1.
K = 0.2
Nsamples = 3000
warmup = 1000
interval = [15, 20, 30, 40, 50, 60, 75] # To be modified
Temp_list = [4.5, 4.0, 3.5, 3.0, 2.75, 2.6, 2.5] # Set temperature
restriciton = [10, 15, 25, 35, 40, 40, 40] # set restriction

for i in range(len(size)):
    print('----------------- Size = %i -------------------'%size[i])
    # We don't train local update samples for L > 80, therefore, we  will use the data at L = 80 as the input for L > 80 training.
    if size[i] > 80:
        eff_param = np.loadtxt('Local_data_fitting_eff_param(n=1)(L=80).dat')
        time1 = time.time()
        Restricted_Temp_descend_opt(Temp_list, size[i], J, K, Nsamples, warmup, interval[i], eff_param, restriction[i])
        time2 = time.time()
        print('################# Size = %i ###################'%size[i])
        print('Size = %i, time = %i'%(size[i], time2-time1))
    else:
        eff_param = np.loadtxt('Local_data_fitting_eff_param(n=1)(L=%i).dat'%size[i])
        time1 = time.time()
        Restricted_Temp_descend_opt(Temp_list, size[i], J, K, Nsamples, warmup, interval[i], eff_param, restriction[i])
        time2 = time.time()
        print('################# Size = %i ###################'%size[i])
        print('Size = %i, time = %i'%(size[i], time2-time1))
        
