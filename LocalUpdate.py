### complete local update algorithm based on Hastings-Metropolis

###import libraries
import numpy.random as rnd
import numpy as np

class LocalUpdator():
    
    def __init__(self, config):
        self.config = config
        L = config.size
        self.J = config.J
        
    def local_update(self):
        i, j = rnd.randint(L, size=(2)) # pick a random site
    
        # compute energy difference
        coef = 2 * config.spins[i,j]
        delta_energy = J * coef * (config.spins[i,(j+1)%self.L] + config.spins[(i+1)%self.L,j] + config.spins[i,(j-1)%self.L] + config.spins[(i-1)%self.L,j]) +\
                        K * coef * (config.spins[i,(j+1)%self.L] * config.spins[(i+1)%self.L,(j+1)%self.L] * config.spins[(i+1)%self.L,j] + \
                                    config.spins[i,(j+1)%self.L] * config.spins[(i-1)%self.L,(j+1)%self.L] * config.spins[(i-1)%self.L,j] + \
                                    config.spins[i,(j-1)%self.L] * config.spins[(i-1)%self.L,(j-1)%self.L] * config.spins[(i-1)%self.L,j] + \
                                    config.spins[i,(j-1)%self.L] * config.spins[(i+1)%self.L,(j-1)%self.L] * config.spins[(i+1)%self.L,j])

        # accept modification with Metropolis probability
        # if not accepted: leave configuration unchanged
        if rnd.random() < np.exp(-self.config.beta * delta_energy):
            self.config.spins[i,j] *= -1
            self.config.energy += delta_energy
            self.config.magnetization += 2*config.spins[i,j]
            
        return config