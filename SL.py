###Train third-nearest interaction Heff from square interaction H

###import libraries
import numpy.random as rnd
import numpy as np
from Hamiltonian import first_NN_interaction, second_NN_interaction, third_NN_interaction
from Configuration import Configuration
from sklearn import linear_model

def LocalUpdate(config):
    """Modify (or not) a configuration with Metropolis algorithm"""

    L = config.size
    J = config.J
    beta = config.beta
    i, j = rnd.randint(L, size=(2))  # pick a random site

    # compute energy difference
    coef = 2 * J * config.spins[i, j]
    delta_energy = coef * (config.spins[i, (j + 1) % L] + config.spins[(i + 1) % L, j] +
                           config.spins[i, (j - 1) % L] + config.spins[(i - 1) % L, j])
    delta_energy += 2 * config.K * ((config.spins[i, j] * config.spins[(i+1)%L, j] *
                                 config.spins[(i+1)%L, (j+1)%L] * config.spins[i, (j+1)%L]) +
                               (config.spins[i, j] * config.spins[(i+1) % L, j] *
                                config.spins[(i+1) % L, (j-1) % L] * config.spins[i, (j-1) % L]) +
                               (config.spins[i, j] * config.spins[(i-1) % L, j] *
                                config.spins[(i-1) % L, (j+1) % L] * config.spins[i, (j+1) % L]) +
                               (config.spins[i, j] * config.spins[(i-1) % L, j] *
                                config.spins[(i-1) % L, (j-1) % L] * config.spins[i, (j-1) % L]))

    # accept modification with Metropolis probability
    # if not accepted: leave configuration unchanged
    if rnd.random() < np.exp(-beta * delta_energy):
        config.spins[i, j] *= -1

def MakeSamples(T, L, J, K, Nstep, Nsample):
    samples = []
    for n in range(Nsample):
        spins = rnd.choice([-1, 1], size=(L, L))  # either +1 or -1
        config = Configuration(spins, L, J, K, T)
        for i in range(Nstep):
            LocalUpdate(config)
        config.energy = config._get_energy()
        E1 = first_NN_interaction(config.spins)
        E2 = second_NN_interaction(config.spins)
        E3 = third_NN_interaction(config.spins)
        Energy = config.energy
        samples.append([E1, E2, E3, Energy])

    return samples

def train(T, L, J, K, Nstep, Nsample):
    # make and save samples
    samples = MakeSamples(T, L, J, K, Nstep, Nsample)
    np.savetxt("samples.txt", samples)
    x = np.array(samples)
    y = x[:, 3]  #coefficiencies of J1 J2 J3
    x = -np.delete(x, 3, axis=1)   #energy items

    #use linermodel to get J1 J2 J3 E0
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    result = reg.coef_
    result = np.append(result, reg.intercept_)

    return result

T = 2.5
L = 50
J = 1.
K = 0.2
Nstep = 200000
Nsample = 10

print(train(T, L, J, K, Nstep, Nsample))

