# This file mainly plot the change of physical quantities (energy/magnetization/specific heat/susceptibility/etc) versus Temperature 
# for Local update method and Wolff update method.

# import library
import matplotlib.pyplot as plt
import numpy as np
# plot local update data
size = [25, 40]
# reload temp_range
temp_range = np.hstack([np.arange(1.5,2.1,0.2), np.arange(2.1,2.3,0.1),np.arange(2.3,2.7,0.05),np.arange(2.7,2.9,0.1), np.arange(2.9,3.5,0.2)])

for L in size:
    energ = np.loadtxt("Local_update_test_energ_%i.dat"%L)
    mag = np.loadtxt("Local_update_test_mag_%i.dat"%L)
    cv = np.loadtxt("Local_update_test_cv_%i.dat"%L)
    chi = np.loadtxt("Local_update_test_chi_%i.dat"%L)
    
    fig = plt.figure()
    plt.plot(temp_range, np.loadtxt("Local_update_test_energ_%i.dat"%L), '-o', label="energy")
    plt.plot(temp_range, np.abs(np.loadtxt("Local_update_test_mag_%i.dat"%L)), '-o', label="magnetization")
    plt.plot(temp_range, np.loadtxt("Local_update_test_cv_%i.dat"%L), '-o', label="specific heat")
    plt.plot(temp_range, np.loadtxt("Local_update_test_chi_%i.dat"%L)/100, '-o', label="susceptibility")
    plt.xlim(1.5, 3.3)
    plt.legend()
    plt.title("Physical quantities", fontsize=25)
    plt.xlabel("$T$", fontsize=20)
    plt.savefig('Local_update_physical_quantities_vs_temp(L = %i).png'%L)
    plt.show()
    
# plot local update data
size = [25, 40]
for L in size:
    energ = np.loadtxt("Wolff_update_test_energ_%i.dat"%L)
    mag = np.loadtxt("Wolff_update_test_mag_%i.dat"%L)
    cv = np.loadtxt("Wolff_update_test_cv_%i.dat"%L)
    chi = np.loadtxt("Wolff_update_test_chi_%i.dat"%L)
    
    fig = plt.figure()
    plt.plot(temp_range, np.loadtxt("Wolff_update_test_energ_%i.dat"%L), '-o', label="energy")
    plt.plot(temp_range, np.abs(np.loadtxt("Wolff_update_test_mag_%i.dat"%L)), '-o', label="magnetization")
    plt.plot(temp_range, np.loadtxt("Wolff_update_test_cv_%i.dat"%L), '-o', label="specific heat")
    plt.plot(temp_range, np.loadtxt("Wolff_update_test_chi_%i.dat"%L)/1000, '-o', label="susceptibility")
    plt.xlim(1.5,3.3)
    plt.legend()
    plt.title("Physical quantities", fontsize=25)
    plt.xlabel("$T$", fontsize=20)
    plt.savefig('Local_update_physical_quantities_vs_temp(L = %i).png'%L)
    plt.show()
