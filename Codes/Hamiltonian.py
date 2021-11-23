# This file consists of functions to calculate Hamiltonian and effective Hamiltonian

#import libraries
import numpy as np
import itertools


def first_NN_interaction(spins):
    """Calculate 1st NN interaction.
    Input: 
    spins: spins configuration matrix.
    Output: 
    sum of 1st nearest neighbour interactions.
    """
    # calculate 1st NN interaction
    ## shift the spins matrix horizontally by one grid
    spins_horiz_shift = np.insert(spins[:,1:], len(spins[0,:])-1, spins[:,0], axis = 1)
        
    ## shift the spins matrix vertically by one grid
    spins_vert_shift = np.insert(spins[1:,:], len(spins[:,0])-1, spins[0,:], axis = 0)
        
    ## calculate \Sigma_{<i,j>} S_i*S_j in the periodic lattice
    interaction_sum = np.sum(spins * spins_horiz_shift + spins * spins_vert_shift)
    return interaction_sum
    
    
def second_NN_interaction(spins):
    """Calculate 2nd NN interaction.
    Input: 
    spins: spins configuration matrix.
    Output: 
    sum of 2nd nearest neighbour interactions.
    """
    # calculate 2nd NN interaction
    # shift the spins matrix by one grid along down right diagonal
    spins_horiz_shift = np.insert(spins[:,1:], len(spins[0,:])-1, spins[:,0], axis = 1)
    spins_dw_right_shift = np.insert(spins_horiz_shift[1:,:], len(spins_horiz_shift[:,0])-1, spins_horiz_shift[0,:], axis = 0)
        
    ## shift the spins matrix by one grid along up right diagonal
    spins_up_right_shift = np.insert(spins_horiz_shift[:-1,:], 0, spins_horiz_shift[-1,:],axis = 0)
        
    interaction_sum = np.sum(spins*spins_dw_right_shift + spins*spins_up_right_shift)
    return interaction_sum
    
def third_NN_interaction(spins):
    """Calculate 3rd NN interaction.
    Input: 
    spins: spins configuration matrix.
    Output: 
    sum of 3rd nearest neighbour interactions.
    """
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
    """Calculate four body term in the periodic lattice
    Input: 
    spins: spins configuration matrix.
    Output:
    sum of four body terms.
    """
    size = len(spins)
    Sum = 0
    for i,j in itertools.product(range(size), repeat=2):
        Sum += spins[i,j] * spins[i,(j+1)%size] * spins[(i+1)%size,j] \
                    * spins[(i+1)%size, (j+1)%size] # only consider the right and top part for each site
    return Sum


def Hamiltonian(J, K, spins):
    """Calcualte total Hamiltonian of the system.
    Input:
    J, K: parameters in the original Hamiltonian.
    spins: spins configuration matrix.
    Output:
    Original Hamitonian of the system.
    """
    ### energy calculated from the total Hamiltonian
    # Define a function to calculate the two-body energy term of the system, given the Hamiltonian: 
    # H0 = -J sum_{<ij>} S_iS_j 
    energ0 = -J * first_NN_interaction(spins)
    ### Define a function to calculate the external sources of energy term of the system, given the Hamiltonian: 
    ### H1 = - K sum_{<ijkl>} S_i S_j S_k S_l
    if K == 0:
        energ1 = 0
    else:
        energ1 = -K * four_body_sum(spins)
        
    energ = energ0 + energ1
    return energ

def Hamiltonian_eff(eff_param, spins):
    """Calculate effective Hamiltonian, given the effective parameters and spins configuration matrix.
    Input: 
    eff_param: effective parameters trained in SLMC method;
    spins: spins configuration matrix.
    Output:
    Effective Hamiltonian of the system.
    """
    order = len(eff_param) - 1
    # first order effective Hamiltonian
    if order == 1:
        return eff_param[0] - eff_param[1] * first_NN_interaction(spins)
    elif order == 2:
        return eff_param[0] - eff_param[1] * first_NN_interaction(spins) - eff_param[2] * second_NN_interaction(spins)
    elif order == 3:
        return eff_param[0] - eff_param[1] * first_NN_interaction(spins) - eff_param[2] * second_NN_interaction(spins) - eff_param[3] * third_NN_interaction(spins)

