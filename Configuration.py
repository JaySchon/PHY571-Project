### build a class for the system

### import libraries
import numpy as np
import numpy.random as rnd
import itertools


class Configuration0:
    """A configuration of Ising spins without external sources of energy."""
    
    def __init__(self, L, J, T):
        self.size = L
        self.J = J
        self.beta = 1./ (T)
        self.spins = rnd.choice([-1,1],size = (L, L))  #create a random spin configuration
        self.energy = self._get_energy()
        self.magnetization = self._get_magentization()
        
    def _get_energy(self):
        """Return the total energy"""
        ### Define a function to calculate the energy of the system, given the Hamiltonian: 
        ### H = -J sum_{<ij>} S_iS_j - K sum_{<ijkl>} S_i S_j S_k S_l
        energ = 0
        ### calculate the two-body sum
        for i,j in itertools.product(range(self.size), repeat=2):
            energ += -self.J * self.spins[i,j] * (self.spins[i,(j+1)%self.size] + self.spins[(i+1)%self.size,j]) # only consider the right and top part for each site

        return energ
    
    def _get_magentization(self):
        """Return the total magnetization"""
        magnet = (np.sum(self.spins))
        return magnet 


class Configuration:
    """A configuration of Ising spins with four-body interaction term."""
    
    def __init__(self, L, J, K, T):
        self.size = L
        self.J = J
        self.K = K
        self.beta = 1./ (T)
        self.spins = rnd.choice([-1,1],size = (L, L))  #create a random spin configuration
        self.energy = self._get_energy()
        self.magnetization = self._get_magnetization()
        
    def get_energy_0(self):
        """Return the two-body energy term"""
        ### Define a function to calculate the two-body energy term of the system, given the Hamiltonian: 
        ### H0 = -J sum_{<ij>} S_iS_j 
        energ0 = 0
        
        ### calculate the two-body sum
        for i,j in itertools.product(range(self.size), repeat=2):
            energ0 += -self.J * self.spins[i,j] * (self.spins[i,(j+1)%self.size] + self.spins[(i+1)%self.size,j]) # only consider the right and top part for each site
        return energ0 
    
    def get_energy_1(self):
        """Return the external sources of energy term"""
        ### Define a function to calculate the external sources of energy term of the system, given the Hamiltonian: 
        ### H1 = - K sum_{<ijkl>} S_i S_j S_k S_l
        energ1 = 0
        
        if self.K == 0:
            return energ1
        else:
            ### calculate the four-body sum
            for i,j in itertools.product(range(self.size), repeat=2):
                energ1 += -self.K * self.spins[i,j] * self.spins[i,(j+1)%self.size] * self.spins[(i+1)%self.size,j] * self.spins[(i+1)%self.size, (j+1)%self.size] # only consider the right and top part for each site
        
            return energ1
    
    def _get_energy(self):
        energ = self.get_energy_0() + self.get_energy_1()
        return energ
    
    def _get_magnetization(self):
        """Return the total magnetization"""
        magnet = (np.sum(self.spins))
        return magnet 
        
