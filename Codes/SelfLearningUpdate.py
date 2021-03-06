# Implement Self-Learning Update Method with effective Hamiltonian and Wolff update algotithm
# No restriction on the cluster

# import libraries
import numpy.random as rnd
import numpy as np
from Hamiltonian import Hamiltonian, Hamiltonian_eff

class SelfLearningUpdate():
    """A class too implement self learning update. There is no restriction in the process of building the cluster."""
    def __init__(self, spins, J, K, T, eff_param):
        self.spins = spins
        self.size = len(spins)
        self.cluster = []
        self.J = J
        self.K = K
        self.beta = 1./ T
        ## parameter extracted from effective Hamiltonian, namely a list with the first term being E0 and next terms being J coefficients
        self.eff_param = eff_param
        ## self.n is the order of interaction that is considered in Hamiltonian_eff
        self.n = len(eff_param) - 1
        
    def activate_prob(self, point, neigh, J):
        """To calculate the activation probability of link."""
        return 1 - np.exp(- 2 * self.beta * J * \
                                  self.spins[point[0],point[1]] * self.spins[neigh[0],neigh[1]])
    
    def add_sites(self, point):
        """To build the cluster from the "point"."""
        ### add sites to the cluster, considering 1st NN, 2nd NN and 3rd NN
        for k in range(self.n):
            for neigh in self.find_NN_neigh(point, k+1):
                if neigh not in self.cluster:
                    prob = self.activate_prob(point, neigh, self.eff_param[k+1])
                    # check if the link is activated
                    if rnd.random() < prob:
                        self.cluster.append(neigh) # add point to the cluster
                        self.add_sites(neigh) # extend the cluster from the center positioned at 'neighbour point'
    
    def find_NN_neigh(self, point, n):
        """To find nth NN neighbours (up to 3)"""
        # initiate neighbour list
        neigh = []
        if n == 1:
            neigh.append([(point[0]+1)%self.size, point[1]])
            neigh.append([point[0], (point[1]+1)%self.size])
            neigh.append([(point[0]-1)%self.size, point[1]])
            neigh.append([point[0], (point[1]-1)%self.size])
        elif n == 2:
            neigh.append([(point[0]+1)%self.size, (point[1]+1)%self.size])
            neigh.append([(point[0]-1)%self.size, (point[1]+1)%self.size])
            neigh.append([(point[0]-1)%self.size, (point[1]-1)%self.size])
            neigh.append([(point[0]+1)%self.size, (point[1]-1)%self.size])
        elif n == 3:
            neigh.append([(point[0]+2)%self.size, point[1]])
            neigh.append([point[0], (point[1]+2)%self.size])
            neigh.append([(point[0]-2)%self.size, point[1]])
            neigh.append([point[0], (point[1]-2)%self.size])
        return neigh
            
    def SLMC_Update(self):
        """To implement Self Learning update, with no restriction in building the cluster."""
        
        E_a = Hamiltonian(self.J, self.K, self.spins)
        #print('E_a:',E_a)
        E_a_eff = Hamiltonian_eff(self.eff_param, self.spins) ### note the definition of hamiltonian_eff !!!
        #print('E_a_eff',E_a_eff)
        #randomly pick a site and add to the cluster
        i, j = rnd.randint(self.size, size=(2)) 
        self.cluster.append([i,j])
        
        #check adjacent states to build the whole cluster, first add 1st NN, then 2nd NN, and then 3rd NN.
        self.add_sites([i,j])
        # flip the spins and calculate parameters.
        for site in self.cluster:
            self.spins[site[0],site[1]] *= -1 
        
        E_b = Hamiltonian(self.J, self.K, self.spins)
        #print('E_b:',E_b)
        E_b_eff = Hamiltonian_eff(self.eff_param, self.spins) 
        #print('E_b_eff',E_b_eff)
        energy_diff = (E_b - E_b_eff) - (E_a - E_a_eff)
        #print('energy_diff:',energy_diff)
        prob = np.min([1, np.exp(- self.beta * energy_diff)])
        #print('prob:',prob)
        # check if we keep the flip
        if rnd.random() < prob:
            #print('flip')
            return self.spins
            
        else: 
            for site in self.cluster:
                self.spins[site[0],site[1]] *= -1
            #print('no flip')
            return self.spins
