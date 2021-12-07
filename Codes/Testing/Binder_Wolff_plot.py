# This file aims to plot binder cumulant vs Temp to determine critical temperature with Wolff update algorithm.

# import libraries
import numpy as np
import matplotlib.pyplot as plt

# Load files
Temp_10 = np.loadtxt('Temp_Wolff(L = 10).dat')
binder_10 = np.loadtxt('Wolff_10.dat')
Temp_25 = np.loadtxt('Temp_Wolff(L = 25).dat')
binder_25 = np.loadtxt('Wolff_25.dat')
Temp_40 = np.loadtxt('Temp_Wolff(L = 40).dat')
binder_40 = np.loadtxt('Wolff_40.dat')
# Plot data
fig = plt.figure(figsize=(8,6))
plt.plot(Temp_10, binder_10, '-o', c = 'red', label="L = 10")
plt.plot(Temp_25, binder_25, '-o', c = 'blue', label="L = 25")
plt.plot(Temp_40,binder_40, '-o',c='green',label = 'L = 40')
plt.title("Determine Critical Temperature", fontsize=25)
plt.xlabel("$Temp$", fontsize=20)
plt.ylabel("Binder Cumulant", fontsize=20)
plt.legend()
plt.ylim(0,1.05)
plt.xlim(1.9,4.5)
plt.axvline(2.50,ls='--',c='black')
plt.tight_layout()
plt.savefig('Binder_vs_Temp(Wolff)(L=25).png')