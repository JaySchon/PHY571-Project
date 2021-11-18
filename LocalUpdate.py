### complete local update algorithm based on Hastings-Metropolis

###import libraries
import numpy.random as rnd
import numpy as np

class LocalUpdate():
    
    def __init__(self, spins, J, K, T):
        self.spins = spins
        self.L = len(spins)
        self.J = J
        self.K = K
        self.beta = 1./ T
         
    def local_update(self):
        i, j = rnd.randint(L, size=(2)) # pick a random site
    
        # compute energy difference
        coef = 2 * spins[i,j]
        delta_energy = J * coef * (self.spins[i,(j+1)%self.L] + self.spins[(i+1)%self.L,j] + self.spins[i,(j-1)%self.L] + self.spins[(i-1)%self.L,j])\
                  + K * coef * (self.spins[i,(j+1)%self.L] * self.spins[(i+1)%self.L,(j+1)%self.L] * self.spins[(i+1)%self.L,j] + \
                           self.spins[i,(j+1)%self.L] * self.spins[(i-1)%self.L,(j+1)%self.L] * self.spins[(i-1)%self.L,j] + \
                           self.spins[i,(j-1)%self.L] * self.spins[(i-1)%self.L,(j-1)%self.L] * self.spins[(i-1)%self.L,j] + \
                           self.spins[i,(j-1)%self.L] * self.spins[(i+1)%self.L,(j-1)%self.L] * self.spins[(i+1)%self.L,j])

        # accept modification with Metropolis probability
        # if not accepted: leave configuration unchanged
        if rnd.random() < np.exp(- self.beta * delta_energy):
            self.spins[i,j] *= -1
            
        return self.spins
