# This file is to calculate autocorrelation function for SLMC global update method
# One notices that the result varies if we only calculate autocorrelation once, therefore we will have an unstable correlation time;
# To solve this problem, we decide to do a collection of calculations and take the average value of correlation time as the final result.


import numpy as np
import numpy.random as rnd
from Configuration import Configuration
from SelfLearningUpdate import SelfLearningUpdate
from scipy.optimize import curve_fit
import time
import sys
sys.setrecursionlimit(3000)


size = [10, 25, 40, 60]
# eff_param obtained from Temp_Descent_Method
eff_param = [[3.63934549, 1.11261356], [18.20677823,  1.10828245], [43.85759863,  1.10707017], [96.84205157,  1.1069041 ]]
J = 1
K = 0.2
T = 2.5
#set repetition number of autocorrelation function calculation
Iter = 30
Ave_Tau =[]

for s in range(len(size)):
    L = size[s]
    tau = []
    for m in range(Iter):
        nt = 500
        O_abs = []
        C = np.zeros(nt)
        n_cycles = 10000
        length_cycle = L**2
        n_warmup = 1000
        spins = rnd.choice([-1,1],size = (L, L))
        # Monte Carlo
        for n in range(n_cycles + n_warmup + nt):

            update = SelfLearningUpdate(spins, J, K, T, eff_param[s])
            spins = update.SLMC_Update()

            if n >= n_warmup:
                config = Configuration(spins, L, J, K, T)
                O_abs.append(abs(config.magnetization))
        O_mean = np.mean(O_abs)
        O_prime = [(i - O_mean ) for i in O_abs]
        O_square = [i**2 for i in O_prime]
        O_square_mean = np.mean(O_square)
        for t in range(nt):
            Sum = 0
            for k in range(n_cycles+nt-t):
                Sum += O_prime[k]*O_prime[k+t]
            Sum /= (n_cycles+nt-t)
            C[t] = Sum / O_square_mean
        np.savetxt('SLMC_correl_function(L=%i)(Iter=%i)'%(L,m),C)
        print('SLMC_correl_function(L=%i)(Iter=%i), done!'%(L,m))
        n_fit_pts = 50
        xr = np.arange(n_fit_pts, dtype=float)
        # fit autocorrelation function
        f = lambda x, a, b: a*np.exp(-x/float(b))
        a, b = curve_fit(f, xr, C[0:n_fit_pts], p0=(1000,1))[0]
        print("Autocorrelation time =", b)
        tau.append(b)
        
    np.savetxt('SLMC_Correlation_time(L = %i):'%L, tau)
    print('Mean_Correlation_time_SLMC_Update(L=%i)'%L,np.mean(tau))
    print('Size %i, done.'%L)
    Ave_Tau.append(np.mean(tau))
    
np.savetxt('Mean_Correlation_Time_SLMC_Updte.dat', Ave_Tau)
np.savetxt('Correlation_Time_Size_SLMC_Update.dat',size)
print('Calculation of Correlation function for SLMC Update Algorithm, done!')