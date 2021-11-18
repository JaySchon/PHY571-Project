# build a class for the 2D Ising system

## import libraries
import numpy as np
import numpy.random as rnd
import itertools


def first_NN_interaction(spins):
    # calculate 1st NN interaction
    ## shift the spins matrix horizontally by one grid
    spins_horiz_shift = np.insert(spins[:,1:], len(spins[0,:])-1, spins[:,0], axis = 1)
        
    ## shift the spins matrix vertically by one grid
    spins_vert_shift = np.insert(spins[1:,:], len(spins[:,0])-1, spins[0,:], axis = 0)
        
    ## calculate \Sigma_{<i,j>} S_i*S_j in the periodic lattice
    interaction_sum = np.sum(spins * spins_horiz_shift + spins * spins_vert_shift)
    return interaction_sum
    
    
def second_NN_interaction(spins):
    # calculate 2nd NN interaction
    # shift the spins matrix by one grid along down right diagonal
    spins_horiz_shift = np.insert(spins[:,1:], len(spins[0,:])-1, spins[:,0], axis = 1)
    spins_dw_right_shift = np.insert(spins_horiz_shift[1:,:], len(spins_horiz_shift[:,0])-1, spins_horiz_shift[0,:], axis = 0)
        
    ## shift the spins matrix by one grid along up right diagonal
    spins_up_right_shift = np.insert(spins_horiz_shift[:-1,:], 0, spins_horiz_shift[-1,:],axis = 0)
        
    interaction_sum = np.sum(spins*spins_dw_right_shift + spins*spins_up_right_shift)
    return interaction_sum
    
def third_NN_interaction(spins):
    # calculate 3rd NN interaction
    ## shift the spins matrix by two grids horizontally
    spins_horiz_shift = np.insert(spins[:,1:], len(spins[0,:])-1, spins[:,0], axis = 1)
    spins_horiz_shift = np.insert(spins_horiz_shift[:,1:], len(spins_horiz_shift[0,:])-1, spins_horiz_shift[:,0], axis = 1)
        
    ## shift the spins matrix by two grids vertically
    spins_vert_shift = np.insert(spins[1:,:], len(spins[:,0])-1, spins[0,:], axis = 0)
    spins_vert_shift = np.insert(spins_vert_shift[1:,:], len(spins_vert_shift[:,0])-1, spins_vert_shift[0,:], axis = 0)
        
    interaction_sum = np.sum(spins*spins_horiz_shift + spins*spins_vert_shift)
    return interaction_sum
    
def four_body_sum(spins):
    """Calculate four bofy term in the periodic lattice"""
    size = len(spins)
    Sum = 0
    for i,j in itertools.product(range(size), repeat=2):
        Sum += spins[i,j] * spins[i,(j+1)%size] * spins[(i+1)%size,j] \
                    * spins[(i+1)%size, (j+1)%size] # only consider the right and top part for each site
    return Sum


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
        ### energy calculated from the total Hamiltonian
        # Define a function to calculate the two-body energy term of the system, given the Hamiltonian: 
        # H0 = -J sum_{<ij>} S_iS_j 
        energ0 = -self.J * first_NN_interaction(self.spins)
        ### Define a function to calculate the external sources of energy term of the system, given the Hamiltonian: 
        ### H1 = - K sum_{<ijkl>} S_i S_j S_k S_l
        if self.K == 0:
            energ1 = 0
        else:
            energ1 = -self.K * four_body_sum(self.spins)
        
        energ = energ0 + energ1
        return energ
    
    def _get_magnetization(self):
        """Return the total magnetization"""
        magnet = np.sum(self.spins)
        return magnet 
        
