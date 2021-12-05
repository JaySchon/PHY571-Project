# This file conists of the functions which are used in SLMC/Restricted-SLMC training.

import numpy as np
import numpy.random as rnd
from sklearn import linear_model
from Configuration import Configuration
from Hamiltonian import first_NN_interaction, second_NN_interaction, third_NN_interaction
from LocalUpdate import LocalUpdate
from SelfLearningUpdate import SelfLearningUpdate
from RestrictedSelfLearningUpdate import RestrictedSelfLearningUpdate

def Make_Samples_Local(size, J, K, T, Nsamples, Nsteps):
    """Generate samples from Local Update Method.
    Input parameters:
    size: size of the lattice;
    J, K: parameters of the Hamiltonian;
    T: temperature;
    Nsamples: the number of the samples;
    Nsteps: steps taken in MC simulation to reach equilibrium;
    Output:
    A list of samples, each sample has the form: [1st NN interaction sum, 2nd NN interaction sum, 3rd NN interaction sum, energy]
    """
    #initiate sample list
    samples = []

    for n in range(Nsamples):
        spins = rnd.choice([-1, 1], size=(size, size))  # either +1 or -1
        for i in range(Nsteps):
            for k in range(size * size):
                update = LocalUpdate(spins, J, K, T)
                spins = update.local_update()
        config = Configuration(spins, size, J, K, T)

        C1 = first_NN_interaction(spins)
        C2 = second_NN_interaction(spins)
        C3 = third_NN_interaction(spins)
        Energy = config.energy
        samples.append([Energy, C1, C2, C3])
        # print info because progress can be slow
        if (n+1) % 100 == 0:
            print("Sample %.0f: %.2f percent done."%(n+1, 100*(n+1)/Nsamples))
    return samples

def Make_tSamples_Local(size, J, K, T, Nsamples, warmup, interval):
    """Generate samples based on local update method, all the samples are taken from one Markov chain.
    Input parameters:
    size: size of the lattice;
    J, K: parameters of the Hamiltonian;
    T: temperature;
    Nsamples: the number of the samples;
    warmup: steps taken in MC simulation to reach equilibrium;
    interval: steps between two cuts of samples;
    Output:
    A list of samples, each sample has the form: [energy, 1st NN interaction sum, 2nd NN interaction sum, 3rd NN interaction sum]
    """
    #initiate sample list
    samples = []
    spins = rnd.choice([-1, 1], size=(size, size))  # either +1 or -1
    
    for n in range(warmup):
        for k in range(size**2):
            update = LocalUpdate(spins, J, K, T)
            spins = update.local_update()
    
    for n in range(Nsamples):
        for i in range(interval):
            for k in range(size**2):
                update = LocalUpdate(spins, J, K, T)
                spins = update.local_update()
        config = Configuration(spins, size, J, K, T)
        C1 = first_NN_interaction(spins)
        C2 = second_NN_interaction(spins)
        C3 = third_NN_interaction(spins)
        Energy = config.energy
        samples.append([Energy, C1, C2, C3])
        # print info because progress can be slow
        if (n+1)%10 == 0: 
            print("Sample %.0f: %.2f percent done."%(n+1, 100*(n+1)/Nsamples))
    return samples

def Make_Samples_SelfLearning(size, J, K, T, Nsamples, Nsteps, eff_param):
    """Generate samples based on self learning update Method.
    Input parameters:
    size: size of the lattice;
    J, K: parameters of the Hamiltonian;
    T: temperature;
    Nsamples: the number of the samples;
    Nstep: steps taken in MC simulation to reach equilibrium;
    eff_param: parameters of effective Hamiltonian.
    Output:
    A list of samples, each sample has the form: [energy, 1st NN interaction sum, 2nd NN interaction sum, 3rd NN interaction sum]
    """
    #initiate sample list
    samples = []

    for n in range(Nsamples):
        spins = rnd.choice([-1, 1], size=(size, size))  # either +1 or -1
        for i in range(Nsteps):
            update = SelfLearningUpdate(spins, J, K, T, eff_param)
            spins = update.SLMC_Update()
        config = Configuration(spins, size, J, K, T)

        C1 = first_NN_interaction(spins)
        C2 = second_NN_interaction(spins)
        C3 = third_NN_interaction(spins)
        Energy = config.energy
        samples.append([Energy, C1, C2, C3])
        # print info because progress can be slow
        if (n+1) % 100 == 0:
            print("Sample %.0f: %.2f percent done."%(n+1, 100*(n+1)/Nsamples))
    return samples

def Make_tSamples_SelfLearning(size, J, K, T, Nsamples, warmup, interval, eff_param):
    """Generate samples based on self learning update method, all the samples are taken from one Markov chain.
    Input parameters:
    size: size of the lattice;
    J, K: parameters of the Hamiltonian;
    T: temperature;
    Nsamples: the number of the samples;
    warmup: steps taken in MC simulation to reach equilibrium;
    interval: steps between two cuts of samples;
    eff_param: parameters of effective Hamiltonian.
    Output:
    A list of samples, each sample has the form: [energy, 1st NN interaction sum, 2nd NN interaction sum, 3rd NN interaction sum]
    """
    #initiate sample list
    samples = []
    spins = rnd.choice([-1, 1], size=(size, size))  # either +1 or -1
    
    for n in range(warmup):
        update = SelfLearningUpdate(spins, J, K, T, eff_param)
        spins = update.SLMC_Update()
    
    for n in range(Nsamples):
        for i in range(interval):
            update = SelfLearningUpdate(spins, J, K, T, eff_param)
            spins = update.SLMC_Update()
        config = Configuration(spins, size, J, K, T)
        C1 = first_NN_interaction(spins)
        C2 = second_NN_interaction(spins)
        C3 = third_NN_interaction(spins)
        Energy = config.energy
        samples.append([Energy, C1, C2, C3])
        # print info because progress can be slow
        if (n+1)%100 == 0: 
            print("Sample %.0f: %.2f percent done."%(n+1, 100*(n+1)/Nsamples))
    return samples


def Make_Samples_Restricted_SelfLearning(size, J, K, T, Nsamples, Nsteps, eff_param, restriction):
    """Generate samples from restricted self learning update Method.
    Input parameters:
    size: size of the lattice;
    J, K: parameters of the Hamiltonian;
    T: temperature;
    Nsamples: the number of the samples;
    Nstep: steps taken in MC simulation to reach equilibrium;
    eff_param: parameters of effective Hamiltonian;
    restriction: restricted size of the cluster.
    Output:
    A list of samples, each sample has the form: [energy, 1st NN interaction sum, 2nd NN interaction sum, 3rd NN interaction sum]
    """
    #initiate sample list
    samples = []

    for n in range(Nsamples):
        spins = rnd.choice([-1, 1], size=(size, size))  # either +1 or -1
        for i in range(Nsteps):
            update = RestrictedSelfLearningUpdate(spins, J, K, T, eff_param, restriction)
            spins = update.Restricted_SLMC_Update()
        config = Configuration(spins, size, J, K, T)

        C1 = first_NN_interaction(spins)
        C2 = second_NN_interaction(spins)
        C3 = third_NN_interaction(spins)
        Energy = config.energy
        samples.append([Energy, C1, C2, C3])
        # print info because progress can be slow
        if (n+1) % 100 == 0:
            print("Sample %.0f: %.2f percent done."%(n+1, 100*(n+1)/Nsamples))
    return samples

def Make_tSamples_Restricted_SelfLearning(size, J, K, T, Nsamples, warmup, interval, eff_param, restriction):
    """Generate samples based on restricted self learning update method, all the samples are taken from one Markov chain.
    Input parameters:
    size: size of the lattice;
    J, K: parameters of the Hamiltonian;
    T: temperature;
    Nsamples: the number of the samples;
    warmup: steps taken in MC simulation to reach equilibrium;
    interval: steps between two cuts of samples;
    eff_param: parameters of effective Hamiltonian.
    restriction: restricted size of the cluster.
    Output:
    A list of samples, each sample has the form: [energy, 1st NN interaction sum, 2nd NN interaction sum, 3rd NN interaction sum]
    """
    #initiate sample list
    samples = []
    spins = rnd.choice([-1, 1], size=(size, size))  # either +1 or -1
    
    for n in range(warmup):
        update = RestrictedSelfLearningUpdate(spins, J, K, T, eff_param, restriction)
        spins = update.Restricted_SLMC_Update()
    
    for n in range(Nsamples):
        for i in range(interval):
            update = RestrictedSelfLearningUpdate(spins, J, K, T, eff_param, restriction)
            spins = update.Restricted_SLMC_Update()
        config = Configuration(spins, size, J, K, T)
        C1 = first_NN_interaction(spins)
        C2 = second_NN_interaction(spins)
        C3 = third_NN_interaction(spins)
        Energy = config.energy
        samples.append([Energy, C1, C2, C3])
        # print info because progress can be slow
        if (n+1)%100 == 0: 
            print("Sample %.0f: %.2f percent done."%(n+1, 100*(n+1)/Nsamples))
    return samples

def train_eff_Hamil(samples, n):
    """Train effective Hamiltonian from the samples.
    Input parameters:
    samples: a list of samples, each sample has the form: [energy, 1st NN interaction sum, 2nd NN interaction sum, 3rd NN interaction sum]
    n: the order of interactions that is considered in H_eff.
    Output:
    eff_param: parameters of effective Hamiltonian, a list with the first term being E0 and next terms being J coefficients
    """
    eff_param = []
    samples = np.array(samples)
    energy = samples[:, 0:1]
    interaction = samples[:,1:n+1]

    #use linear model to get E0 and Js
    reg = linear_model.LinearRegression()
    reg.fit(interaction, energy)

    coef = reg.coef_
    eff_param = np.append(reg.intercept_, -coef)
    return eff_param


def self_optimization(Iter, size, J, K, T, Nsamples, warmup, interval, eff_param):
    """
    To complete the self optimization procedure in the self-learning Monte Carlo.
    Input: 
    Iter: the number of iteration steps to optimize effective Hamiltonian;
    size: size of the lattice;
    J, K: parameters of the Hamiltonian;
    T: temperature;
    Nsamples: the number of the samples;
    warmup: steps taken in MC simulation to reach equilibrium;
    interval: steps between two cuts of samples;
    eff_param: effective Hamiltonian obtained from local update monte carlo at T > Tc.
    Output: 
    optimized parameters of effective Hamiltonian.
    """
    # the order of interactions in effective Hamiltonian
    n = len(eff_param) - 1
    
    for k in range(Iter):
        # for every iteration,
        # create Nsamples samples with the eff_param obtained from the last step
        samples = Make_tSamples_SelfLearning(size, J, K, T, Nsamples, warmup, interval, eff_param)
        # use the samples to train new eff_param
        eff_param = train_eff_Hamil(samples, n)
        print('Iteration %.0f, %.2f percent done.'%(k+1, 100*(k+1)/Iter))
        print('eff_param is:', eff_param)
    return eff_param

def Temp_descend_opt(Temp, size, J, K, Nsamples, warmup, interval, eff_param):
    """
    To complete the self optimization procedure in the self-learning Monte Carlo.
    Input: 
    Temp: a list of temperature, namely the way how temperature is descending.
    size: size of the lattice;
    J, K: parameters of the Hamiltonian;
    Nsamples: the number of the samples trained at each temperature;
    warmup: steps taken in MC simulation to reach equilibrium;
    interval: steps between two cuts of samples;
    eff_param: effective Hamiltonian obtained from local update monte carlo at T > Tc.
    Return: 
    optimized parameters of effective Hamiltonian at the last temperature.
    Output: 
    intermediate temp and corresponding eff_param.
    """
    # the order of interactions in effective Hamiltonian
    n = len(eff_param) - 1
    
    for i in range(len(Temp)):
        samples = Make_tSamples_SelfLearning(size, J, K, Temp[i], Nsamples, warmup, interval, eff_param)
        eff_param = train_eff_Hamil(samples, n)
        print('Temp %.3f, %.2f percent done.'%(Temp[i], 100*(i+1)/len(Temp)))
        print('eff_param is:', eff_param)
    return eff_param
    
def restricted_self_optimization(Iter, size, J, K, T, Nsamples, warmup, interval, eff_param, restriction):
    """
    To complete the self optimization procedure in the restricted self-learning Monte Carlo.
    Input: 
    Iter: the number of iteration steps to optimize effective Hamiltonian;
    size: size of the lattice;
    J, K: parameters of the Hamiltonian;
    T: temperature;
    Nsamples: the number of the samples;
    warmup: steps taken in MC simulation to reach equilibrium;
    interval: steps between two cuts of samples;
    eff_param: effective Hamiltonian obtained from local update monte carlo at T > Tc;
    restriction: restricted size of the cluster.
    Output: 
    optimized parameters of effective Hamiltonian.
    """
    # the order of interactions in effective Hamiltonian
    n = len(eff_param) - 1
    
    for k in range(Iter):
        # for every iteration,
        # create Nsamples samples with the eff_param obtained from the last step
        samples = Make_tSamples_Restricted_SelfLearning(size, J, K, T, Nsamples, warmup, interval, eff_param, restriction)
        # use the samples to train new eff_param
        eff_param = train_eff_Hamil(samples, n)
        print('Iteration %.0f, %.2f percent done.'%(k+1, 100*(k+1)/Iter))
        print('eff_param is', eff_param)
    return eff_param

def Restricted_Temp_descend_opt(Temp, size, J, K, Nsamples, warmup, interval, eff_param, restriction):
    """
    To complete the self optimization procedure in the self-learning Monte Carlo.
    Input: 
    Temp: a list of temperature, namely the way how temperature is descending.
    size: size of the lattice;
    J, K: parameters of the Hamiltonian;
    Nsamples: the number of the samples trained at each temperature;
    warmup: steps taken in MC simulation to reach equilibrium;
    interval: steps between two cuts of samples;
    eff_param: effective Hamiltonian obtained from local update monte carlo at T > Tc;
    restriction: restricted size of the cluster.
    Return: 
    optimized parameters of effective Hamiltonian at the last temperature.
    Output: 
    intermediate temp and corresponding eff_param.
    """
    # the order of interactions in effective Hamiltonian
    n = len(eff_param) - 1
    
    for i in range(len(Temp)):
        samples = Make_tSamples_Restricted_SelfLearning(size, J, K, Temp[i], Nsamples, warmup, interval, eff_param, restriction)
        eff_param = train_eff_Hamil(samples, n)
        print('Temp %.3f, %.2f percent done.'%(Temp[i], 100*(i+1)/len(Temp)))
        print('eff_param is:', eff_param)
    return eff_param