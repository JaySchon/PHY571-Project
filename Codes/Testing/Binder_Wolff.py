# This file is aimed to calculate Critical Temperature of Modified Ising Model with Binder Cumulant method based on Wolff update algorithm.

# import libraries
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from Configuration import Configuration
from WolffUpdate import WolffUpdate
from LocalUpdate import LocalUpdate

# Define functions to calculate binder cumulant.
def fourth_order_moment(lst):
    """Calculate the fourth order moment of an array.
    Input:
    lst: a list or an ndarray.
    Output:
    A number: fourth order moment of the data in the list.
    """
    mean = np.mean(lst)
    return np.sum((lst - mean)**4)/len(lst)
    
    
def binder_cumulant(J, K, size, Temp, Nsamples, interval, warmup, method):
    """Calculate binder_cumulant at given temperature for Local update method or Wolff update method.
    Input parameters:
    J, K: parameters of the Hamiltonian;
    size: configuration size;
    Temp: temperature;
    Nsamples: the number of the samples;
    interval: steps between two cuts of samples;
    warmup: steps taken in MC simulation to reach equilibrium;
    method: update method, options are 'Local' or 'Wolff'.
    Output:
    binder cumulant value.
    """
    spins = rnd.choice([-1, 1], size=(size, size))
    if method == 'Wolff':
        samples = []
        for n in range(warmup):
            update = WolffUpdate(spins, J, K, Temp)
            spins = update.Wolff_Update_1()
        for n in range(Nsamples):
            for i in range(interval):
                update = WolffUpdate(spins, J, K, Temp)
                spins = update.Wolff_Update_1() 
            config = Configuration(spins, size, J, K, Temp)
            # generate a sample of magnetization
            samples.append(config.magnetization)
    elif method == 'Local':
        samples = []
        length_cycle = size ** 2
        for n in range(warmup):
            for k in range(length_cycle):
                update = LocalUpdate(spins, J, K, Temp)
                spins = update.local_update()
        # Generate samples from Markov chain
        for n in range(Nsamples):
            for i in range(interval):
                for k in range(length_cycle):
                    update = LocalUpdate(spins, J, K, Temp)
                    spins = update.local_update() 
            config = Configuration(spins, size, J, K, Temp)
            # generate a sample of magnetization
            samples.append(config.magnetization)
    else:
        return ('Please check the method that you input! Options available are: "Local" and "Wolff". ')
    second_moment = np.var(samples)
    fourth_moment = fourth_order_moment(samples)
    return 1.5 * (1 - fourth_moment / (3 * second_moment**2))


# Calculate Binder cumulant versus temperature using Wolff update
# This is a production script.

J = 1.
K = 0.2
size = [10, 25, 40]
Nsamples_1 = 1000
Nsamples_2 = 2000
Nsamples_3 = 3000
Nsamples_4 = 4000
# Set temperature interval
Temp_Wolff_10 = np.hstack([np.arange(2.0,2.3,0.1), np.arange(2.3,2.7,0.025),np.arange(2.75,2.9,0.05),np.arange(2.9,3.3,0.1),np.arange(3.3,4.7,0.2)])
Temp_Wolff_25 = np.hstack([np.arange(2.0,2.3,0.1), np.arange(2.3,2.7,0.025),np.arange(2.75,2.9,0.05),np.arange(2.9,3.3,0.1),np.arange(3.3,4.1,0.2)])
Temp_Wolff_40 = np.hstack([np.arange(2.0,2.3,0.1), np.arange(2.3,2.7,0.025),np.arange(2.75,2.9,0.05),np.arange(2.9,3.3,0.1)])
# 
interval = [10, 12, 25]
warmup = 1000
method = 'Wolff'

np.savetxt('Temp_Wolff(L = 10).dat',Temp_Wolff_10)
np.savetxt('Temp_Wolff(L = 25).dat',Temp_Wolff_25)
np.savetxt('Temp_Wolff(L = 40).dat',Temp_Wolff_40)
for i in range(len(size)):
    Temp = np.loadtxt('Temp_Wolff(L = %i).dat'%size[i])
    binder = np.zeros(len(Temp))
    if size[i] == 10:
        for j in range(len(Temp)):
            if Temp[j] < 2.8:
                binder[j] = binder_cumulant(J, K, size[i], Temp[j], Nsamples_1, interval[i], warmup, method)
            elif Temp[j] >= 2.8:
                binder[j] = binder_cumulant(J, K, size[i], Temp[j], Nsamples_2, interval[i], warmup, method)
            print('Size = %i, Temp = %.3f, done.'%(size[i],Temp[j]))
        np.savetxt('binder_cumulant_Wolff(L = %i).dat'%size[i], binder)
        
    if size[i] == 25:
        for j in range(len(Temp)):
            if Temp[j] < 2.8:
                binder[j] = binder_cumulant(J, K, size[i], Temp[j], Nsamples_2, interval[i], warmup, method)
            elif Temp[j] >= 2.8:
                binder[j] = binder_cumulant(J, K, size[i], Temp[j], Nsamples_3, interval[i], warmup, method)
            print('Size = %i, Temp = %.3f, done.'%(size[i],Temp[j]))
        np.savetxt('binder_cumulant_Wolff(L = %i).dat'%size[i], binder)

    if size[i] == 40:
        for j in range(len(Temp)):
            if Temp[j] < 2.8:
                binder[j] = binder_cumulant(J, K, size[i], Temp[j], Nsamples_3, interval[i], warmup, method)
            elif Temp[j] >= 2.8:
                binder[j] = binder_cumulant(J, K, size[i], Temp[j], Nsamples_4, interval[i], warmup, method)
            print('Size = %i, Temp = %.3f, done.'%(size[i],Temp[j]))
        np.savetxt('binder_cumulant_Wolff(L = %i).dat'%size[i], binder)        
           
print('Wolff update: Binder cumulant calculation done !')