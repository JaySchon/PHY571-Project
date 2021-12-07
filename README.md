# PHY571 Project

## Self learning Monte Carlo Method

Reproduce the results by referring to the paper  [Self-Learning Monte Carlo Method](https://arxiv.org/abs/1610.03137).

###  Folders

* In the codes folder, there are well wrapped "class" files or ''function'' files, which are called in the calculation tasks.

```
(1) Configuration.py: A class file to construct the modified 2D Ising model;
(2) Hamiltonian.py: A function file to calculate the Hamiltonian or effective Hamiltonian in the project;
(3) LocalUpdate.py: A class file to Complete local update algorithm based on Hastings-Metropolis method;
(4) WolffUpdate.py: A class file to implement Wolff cluster update algorithm for the system with original Hamiltonian;
(5) SelfLearningUpdate.py: A class file to implement Self-Learning Update Method with effective Hamiltonian and Wolff update algotithm;
(6) RestrictedSelfLearningUpdate.py: A class file to implement Self-Learning Update Method with cluster size restriction;
(7) SLMC_Training_Lib.py: A function file which will be used in the SLMC/RSLMC training process, including making samples, self-optimizing, etc.
```

The four update class will take spins configuration as an input parameter, and return the spins configuration after one update.

* In the testing folder, one will find some sub-folders which aims to realize one specific calculation task. Plotting and production files are strictly separated.
* In the data folder, calculation results (including plot files and maybe data files) are presented in this folder.

### Calculation Functions

* **Physical_quantities_vs_Temp**

To calculate how the physical quantities (which we are interested in, including energy, magnetization, specific heat capacity and susceptibility) vary with temperature using Local Update method and Wolff cluster Update method.  

The definition of specific heat capacity is $$C_v = \beta N(\left<m^2\right>-\left<|m|\right>^2)$$, while the definition of susceptibility is $\chi = k_B \beta^2 N(\left<e^2\right>-\left<e\right>^2)$. In all the calculation, we set $k_B = 1$.

To run the file and get calculation results, one should put Physical_Quantities_vs_Temp_plotting.py and Physical_Quantities_vs_Temp_training.py in the codes folder and use the command:

```
>> python Physical_Quantities_vs_Temp_training.py
>> python Physical_Quantities_vs_Temp_plotting.py
```

Training process will take some time, and one should be patient with Monte Carlo Simulation Calculations.

Of course, one is welcome to change parameters set in the file and explore more about the model.

* **Binder Cumulant vs Temp**

The definition of Binder cumulant is: $U(T,C) = \frac{3}{2}(1-\frac{\left<M^4\right>}{3\left<M^2\right>})$. It can be used to determine the critical temperature of Ising model, as suggested in the literature.

To calculate Binder cumulant vs temperature, one should first run "Binder_Wolff" file:

```
>> python Binder_Wolff.py
```

This file will produce some .dat files, which are used in the plotting. For convenience, we put our calculaton results in the folder. If one doesn't want to spend a long time waiting for calculation, he/she can run direclt:

```
>> python Binder_Wolff_plot.py
```

Then the graph will be saved in the same folder. **One should note that you must put both .py files in the codes folder, and it is the same with the following tasks.**

* **SLMC & RSLMC Training**

To complete SLMC or RSLMC training and obtain optimized effective Hamiltonian at T = $T_c$, one should first generate some samples at T larger than $T_c$ using Local Update Algorithm. Run with the command:

```
>> python Local_samples_training.py
```

After a long time of samples production, one will have effective parameters ($J$ as defined in the literature) written in the .dat files. In the file, we choose to generate Local Update samples of size = 10, 25, 40, 60, 80, however, one can have their own choice for this parameter. Correspondingly, one should change "interval" value in the calculation. Because different configuration sizes have different correlation time. 

After that, one can run SLMC_training. There are two approaches for the training, as introduced in our report. One is "ensemble average", the other is "temperature descent". Usually, temperature descent method will cost less time than the other.

To run the first option, one uses the command:

```
>> python SLMC_Training_Ensemble_Average.py
```

To run the second option, one uses the command:

```
>> python SLMC_Training_Temp_Descent.py
```

Please note that in either case, calculated result will be printed and displayed in your terminal instead of written into data files. **Therefore, one is encouraged to save output results by yourself.** This can be done easily, for example, in Linux system, one uses the command:

```
>> python SLMC_Training_Ensemble_Average.py |tee result.txt
```

Moreover, you have to do data dealing by writing a job script by yourself. For example, you have to calculate the average value of effective parameters in ensemble average method.

And the same procedure for RSLMC training.

Our training process is done using only the first oder expansion, namely, we only keep $J_1$ in the effective Hamiltonian. If one wants to try third order expansion, simply change this line in the file

```
eff_param = np.loadtxt('Local_data_fitting_eff_param(n=1)(L=%i).dat'%size[i])
```

to the following:

```
eff_param = np.loadtxt('Local_data_fitting_eff_param(n=3)(L=%i).dat'%size[i])
```

Third order Hamiltonian parameters will be read and training is done by keeping all these three parameters.

* **Acceptance probability**

If one is interested in the flipping probability when construction the clusters in SLMC/RSLMC update method, run with the command:

```
>> python Acceptance_prob_SLMC_RSLMC.py
```

In this file, we calculate flipping probability for SLMC with size 25, 40, 60 and for RSLMC with size 25, 40, 60, 80, 100, 120. One is free to change the setting, however, correspondingly you have to change eff_param which is obtained from (R)SLMC training process and "restriction value" for RSLMC.

The calculated result will be displayed in the terminal, you have to save it by yourself.

* **Autocorrelation function**

The definition is autocorrelation function is given in the original literature. Here, we introduce a normalized coefficient, which will not change the value of correlation time. Let $O_i = |M_i|-\left<|M|\right>$, and then autocorrelation function of $|M|$ is given by $C(\tau)=\frac{\frac{1}{N-t}\sum_iO_iO_{i+t}}{\left<O^2\right>}\propto e^{-\tau/\tau_{correl}}$.

In our experiment, we find that the fitting correlation time varies if we only do one calculation of the autocorrelation function. Therefore, we choose to do a collection of calculations (for example, iter = 30) and take the average value of correlation time as the final result. 

To calculate autocorrelation function and correlation time, one run the command:

```
>> python Autocorrelation_Local.py
>> python Autocorrelation_Wolff.py
>> python Autocorrelation_SLMC.py
>> python Autocorrelation_RSLMC.py
```

After completing the calculation, one open the jupyter notebook file and run each cell. 

We plot Correlation Time vs Size for each update method. We also plot the autocorrelation function for each update method.

### At the end

One is always welcome to do more test with the codes and give suggestions or feedback to us !!!
