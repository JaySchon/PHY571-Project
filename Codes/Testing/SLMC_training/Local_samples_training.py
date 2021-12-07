# This file is aimed to create samples from Local Update Algorithm at T = 5.
# Then, these data are used to train eff_param (order = 1 / order = 3) and generate eff_param.dat files.
# Then, Eff_param are used for SLMC/RSLMC training.

# import libraries
import numpy as np
from SLMC_Training_Lib import Make_tSamples_Local
import time

# Generate Local update data
# T > Tc, train some samples using Local Update Method, T = 5
size = [10, 25, 40, 60, 80]
J = 1.
K = 0.2
T = 5
Nsamples = 50 # modified later
warmup = 1000
interval = [10,40,100,200,400]
# trail number, modified later
for i in range(len(size)):
    time1 = time.time()
    samples = Make_tSamples_Local(size[i], J, K, T, Nsamples, warmup, interval[i])
    np.savetxt('(R)SLMC_training_samples(T=%i, L=%i).dat'%(T, size[i]), samples)
    time2 = time.time()
    print('Samples created for size %i, time = %i'%(size[i],(time2-time1)))

    
# Train eff_param with the samples created above
for L in size:
    samples = np.loadtxt('(R)SLMC_training_samples(T=%i, L=%i).dat'%(T, L))
    n = 1
    # order = 1 eff_param
    eff_param_1 = train_eff_Hamil(samples, n)
    m = 3
    # order = 3 eff_param
    eff_param_2 = train_eff_Hamil(samples, m)
    np.savetxt('Local_data_fitting_eff_param(n=1)(L=%i).dat'%L,eff_param_1)
    np.savetxt('Local_data_fitting_eff_param(n=3)(L=%i).dat'%L,eff_param_2)

print('Eff_params are calculated and written !')
print('Task done !')




