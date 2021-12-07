# This file mainly calculate physical quantities (energy/magnetization/specific heat/susceptibility/etc) versus Temperature 
# for Local update method and Wolff update method. 

# import libraries
import numpy as np
import numpy.random as rnd
from Configuration import Configuration
import Hamiltonian
from LocalUpdate import LocalUpdate
from WolffUpdate import WolffUpdate
import itertools
import time

# Local update
# Training local update data and save in files
# This is a production script, it will save the results in files
size = [25, 40]
J = 1
K = 0.2

# set temperature range
temp_range = np.hstack([np.arange(1.5,2.1,0.2), np.arange(2.1,2.3,0.1),np.arange(2.3,2.7,0.05),np.arange(2.7,2.9,0.1), np.arange(2.9,3.5,0.2)])
mag = np.zeros_like(temp_range)
energ = np.zeros_like(temp_range)
chi = np.zeros_like(temp_range)
cv = np.zeros_like(temp_range)

time1 = time.time()

for L in size:
    spins = rnd.choice([-1,1],size = (L, L))
    for i,T in enumerate(temp_range):
    
        av_m, av_m2, av_e, av_e2 = 0,0,0,0

        n_cycles = 20000
        length_cycle = L*L
        n_warmup = 1000

        # Monte Carlo
        for n in range(n_warmup+n_cycles):
            for k in range(length_cycle):
                update = LocalUpdate(spins, J, K, T)
                spins = update.local_update()
                
            if n >= n_warmup:
                config = Configuration(spins, L, J, K, T)
                av_e  += config.energy
                av_e2 += config.energy**2
                av_m  += abs(config.magnetization)
                av_m2 += config.magnetization**2
            
        # normalize averages
        av_m  /= float(n_cycles)
        av_m2 /= float(n_cycles)
        av_e  /= float(n_cycles)
        av_e2 /= float(n_cycles)
            
        # get physical quantities
        fact = 1./ L**2
        mag[i] = fact * av_m
        energ[i] = fact * av_e
        cv[i] = fact * (av_e2 - av_e**2) / T**2
        chi[i] = fact * (av_m2 - av_m**2) / T
    
        # print info because progress can be slow
        print("T = %f and %.2f percent done"%(T, (100.*(i+1))/len(temp_range)))

    # save quantities in a file
    np.savetxt("Local_update_test_energ_%i.dat"%L, energ)
    np.savetxt("Local_update_test_mag_%i.dat"%L, mag)
    np.savetxt("Local_update_test_cv_%i.dat"%L, cv)
    np.savetxt("Local_update_test_chi_%i.dat"%L, chi)
    
time2 = time.time()
print('Local_update_time: %i'%(time2-time1))

# Wolff update
# Training Wolff update data and save in files
# This is a production script, it will save the results in files

J = 1
K = 0.2
size = [25, 40]

# set temperature range
temp_range = np.hstack([np.arange(1.5,2.1,0.2), np.arange(2.1,2.3,0.1),np.arange(2.3,2.7,0.05),np.arange(2.7,2.9,0.1), np.arange(2.9,3.5,0.2)])
mag = np.zeros_like(temp_range)
energ = np.zeros_like(temp_range)
chi = np.zeros_like(temp_range)
cv = np.zeros_like(temp_range)

time1 = time.time()
# lattice size

for L in size:
    spins = rnd.choice([-1,1],size = (L, L))
    for i,T in enumerate(temp_range):
    
        av_m, av_m2, av_e, av_e2 = 0,0,0,0

        n_cycles = 20000
        n_warmup = 1000

        # Monte Carlo
        for n in range(n_warmup+n_cycles):
            update = WolffUpdate(spins, J, K, T)
            spins = update.Wolff_Update_1()
                
            if n >= n_warmup:
                config = Configuration(spins, L, J, K, T)
                av_e  += config.energy
                av_e2 += config.energy**2
                av_m  += abs(config.magnetization)
                av_m2 += config.magnetization**2
            
        # normalize averages
        av_m  /= float(n_cycles)
        av_m2 /= float(n_cycles)
        av_e  /= float(n_cycles)
        av_e2 /= float(n_cycles)
            
        # get physical quantities
        fact = 1./L**2
        mag[i] = fact * av_m
        energ[i] = fact * av_e
        cv[i] = fact * (av_e2 - av_e**2) / T**2
        chi[i] = fact * (av_m2 - av_m**2) / T
    
        # print info because progress can be slow
        print("T = %f and %.2f percent done"%(T, (100.*(i+1))/len(temp_range)))

    # save quantities in a file
    np.savetxt("Wolff_update_test_energ_%i.dat"%L, energ)
    np.savetxt("Wolff_update_test_mag_%i.dat"%L, mag)
    np.savetxt("Wolff_update_test_cv_%i.dat"%L, cv)
    np.savetxt("Wolff_update_test_chi_%i.dat"%L, chi)
    
time2 = time.time()
print('Wolff_update_time: %i'%(time2-time1))
