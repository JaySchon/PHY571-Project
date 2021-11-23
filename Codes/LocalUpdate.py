# Complete local update algorithm based on Hastings-Metropolis method

## import libraries
import numpy.random as rnd
import numpy as np

class LocalUpdate():
    """A class to implement Local Update method."""
    def __init__(self, spins, J, K, T):
        """
        Initiate parameters.
        spins: a spin configuration matrix;
        J, K: parameters in the Hamiltonian;
        T: temperature.
        """
        self.spins = spins
        self.L = len(spins)
        self.J = J
        self.K = K
        self.beta = 1./ T
         
    def local_update(self):
        """Complete one local update. 
        Output: the spin configuration matrix after one update.
        """
        i, j = rnd.randint(self.L, size=(2)) # pick a random site
    
        # compute energy difference
        coef = 2 * self.spins[i,j]
        delta_energy = self.J * coef * \
                  (self.spins[i,(j+1)%self.L] + self.spins[(i+1)%self.L,j] + self.spins[i,(j-1)%self.L] + self.spins[(i-1)%self.L,j])\
                 + self.K * coef *\
                           (self.spins[i,(j+1)%self.L] * self.spins[(i+1)%self.L,(j+1)%self.L] * self.spins[(i+1)%self.L,j] + \
                           self.spins[i,(j+1)%self.L] * self.spins[(i-1)%self.L,(j+1)%self.L] * self.spins[(i-1)%self.L,j] + \
                           self.spins[i,(j-1)%self.L] * self.spins[(i-1)%self.L,(j-1)%self.L] * self.spins[(i-1)%self.L,j] + \
                           self.spins[i,(j-1)%self.L] * self.spins[(i+1)%self.L,(j-1)%self.L] * self.spins[(i+1)%self.L,j])

        # accept modification with Metropolis probability
        # if not accepted: leave configuration unchanged
        if rnd.random() < np.exp(-self.beta * delta_energy):
            self.spins[i,j] *= -1
            
        return self.spins