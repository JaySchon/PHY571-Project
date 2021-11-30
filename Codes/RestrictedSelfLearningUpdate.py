# This file is to implement Self-Learning Update Method with effective Hamiltonian and Wolff update algotithm
# Add restriction of cluster size defined by Manhattan Distance

# import libraries
import numpy.random as rnd
import numpy as np
from Hamiltonian import Hamiltonian, Hamiltonian_eff

class RestrictedSelfLearningUpdate():
    """A class to implement restrcited self learning update method."""
    def __init__(self, spins, J, K, T, eff_param, restriction):
        ## restriction is a finite number, it gives the size restriction of the cluster
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
        ## initiate a list to include all the spin pairs which is blocked from linking beccause of the size restriction of the cluster
        ## the list include n components, the first corresponds to 1st NN spin pairs blocked, the second corresponds to 2nd NN spin pairs and so on. 
        self.restricted_collection = [[]] * (len(eff_param) - 1)
        self.restriction = restriction
        
    def Manhattan_Dist(self, site1, site2):
        """To calculate Manhattan distance between two spin states, considering periodic boundary condition."""
        x_offset = min(np.abs(site1[0]-site2[0]),self.size - np.abs(site1[0]-site2[0]))
        y_offset = min(np.abs(site1[1]-site2[1]),self.size - np.abs(site1[1]-site2[1]))
        return x_offset + y_offset
    
    def _is_in_cluster(self, ini_site, site):
        """To check if the site is in the restricted cluster."""
        if self.Manhattan_Dist(ini_site, site) > self.restriction:
            return False
        else:
            return True
    
    def activate_prob(self, point, neigh, J):
        """To calculate the activate probavility of the link."""
        return 1 - np.exp(- 2 * self.beta * J * \
                                  self.spins[point[0],point[1]] * self.spins[neigh[0],neigh[1]])
    
    def add_sites(self, point, ini_site):
        """To build the cluster from "point"."""
        ### add sites to the cluster, considering 1st NN, 2nd NN and 3rd NN.
        ### the cluster size is restricted.
        for k in range(self.n):
            for neigh in self.find_NN_neigh(point, k+1):
                if self._is_in_cluster(ini_site, neigh) == True:
                    if neigh not in self.cluster:
                        prob = self.activate_prob(point, neigh, self.eff_param[k+1])
                        # check if the link is activated
                        if rnd.random() < prob:
                            self.cluster.append(neigh) # add point to the cluster
                            self.add_sites(neigh, ini_site) # extend the cluster from the center positioned at 'neighbour point'
                else:
                    ### add blocked pairs into the collection
                    prob = self.activate_prob(point, neigh, self.eff_param[k+1])
                    if prob > 0:
                        self.restricted_collection[k].append((point, neigh))
                        
    def find_NN_neigh(self, point, n):
        """To find nth NN neighbours"""
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
  
    def Restricted_SLMC_Update(self):
        """To implement one restricted self mearning update."""
        ### restricted self learning update
        
        E_a = Hamiltonian(self.J, self.K, self.spins)
        E_a_eff = Hamiltonian_eff(self.eff_param, self.spins) ### note the definition of hamiltonian_eff !!!
        #randomly pick a site and add to the cluster
        i, j = rnd.randint(self.size, size=(2)) 
        self.cluster.append([i,j])
        
        # check adjacent states to build the whole cluster, first add 1st NN, then 2nd NN, and then 3rd NN.
        # add blocked pairs to the restricted_collection
        self.add_sites([i,j], [i,j])
        
        ### calcualte boundary_coeffecient
        ### Expression is written as \Pi_{k} \Pi_{<i,j> \in Restr_k} exp(2 * \beta * \tilde{J_k} * S_ik * S_jk)
        Bound_coeff = 1
        for k in range(self.n):
            Sum = 0
            for (site1, site2) in self.restricted_collection[k]:
                Sum += self.spins[site1[0],site1[1]] * self.spins[site2[0],site2[1]]
            Bound_coeff *= np.exp(2* self.beta * self.eff_param[k+1] * Sum)
        # flip the cluster and calculate energy difference
        for site in self.cluster:
            self.spins[site[0],site[1]] *= -1 
           
        E_b = Hamiltonian(self.J, self.K, self.spins)
        E_b_eff = Hamiltonian_eff(self.eff_param, self.spins) 
        energy_diff = (E_b - E_b_eff) - (E_a - E_a_eff)
        prob = np.min([1, np.exp(- self.beta * energy_diff) * Bound_coeff])
        # check if we keep the flip
        if rnd.random() < prob:
            return self.spins 
            
        else: 
            for site in self.cluster:
                self.spins[site[0],site[1]] *= -1
            return self.spins
            
            