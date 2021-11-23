import numpy as np
import numpy.random as rnd
from Configuration import Configuration
from Hamiltonian import first_NN_interaction, second_NN_interaction, third_NN_interaction
from LocalUpdate import LocalUpdate
from SelfLearningUpdate import SelfLearningUpdate
from sklearn import linear_model

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

        E1 = first_NN_interaction(spins)
        E2 = second_NN_interaction(spins)
        E3 = third_NN_interaction(spins)
        Energy = config.energy
        samples.append([Energy, E1, E2, E3])
        # print info because progress can be slow
        print("Sample %.0f: %.2f percent done."%(n+1, 100*(n+1)/Nsamples))
    return samples


def Make_Samples_SelfLearning(size, J, K, T, Nsamples, Nsteps, eff_param):
    """Generate samples from Local Update Method.
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

        E1 = first_NN_interaction(spins)
        E2 = second_NN_interaction(spins)
        E3 = third_NN_interaction(spins)
        Energy = config.energy
        samples.append([Energy, E1, E2, E3])
        # print info because progress can be slow
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

def eff_paramLocal(L, J, K, T, n, Nsamples, Nsteps):
        # T > Tc, train some samples using Local Update Method, T = 5
    samples = Make_Samples_Local(L, J, K, T, Nsamples, Nsteps)
    print(samples)
    eff_param = train_eff_Hamil(samples, n)
    print(eff_param)
    return eff_param

def Optimizing(L, J, K, T, Nsamples, Nsteps, n, eff_param):
    # Set iteration step
    Iter = 5  # modified later
    for k in range(Iter):
        samples = Make_Samples_SelfLearning(L, J, K, T, Nsamples, Nsteps, eff_param)
        eff_param = train_eff_Hamil(samples, n)
        print(eff_param)

    return eff_param
