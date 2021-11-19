### build a class for the 2D-Ising Model system

### import libraries
import numpy as np
from Hamiltonian import Hamiltonian
    
class Configuration:
    """A configuration of Ising spins with four-body interaction term."""
    
    def __init__(self, spins, L, J, K, T):
        self.size = L
        self.J = J
        self.K = K
        self.beta = 1./ T
        self.spins = spins
        self.energy = self._get_energy()
        self.magnetization = self._get_magnetization()

    def _get_energy(self):
        return Hamiltonian(self.J, self.K, self.spins)
    
    def _get_magnetization(self):
        """Return the total magnetization"""
        magnet = np.sum(self.spins)
        return magnet 
        
