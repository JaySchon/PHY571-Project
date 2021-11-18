# Implement Self-Learning Update Method with effective Hamiltonian and Wolff update algotithm
# No restriction on the cluster

# import libraries
import numpy.random as rnd
import numpy as np
from Hamiltonian import Hamiltonian, Hamiltonian_eff_1, Hamiltonian_eff_2, Hamiltonian_eff_3

class SelfLearningUpdate():
    
    def __init__(self, spins, J, K, T, eff_param):
        self.spins = spins
        self.size = len(spins)
        self.cluster = []
        self.J = J
        self.K = K
        self.beta = 1./ T
        ## parameter extracted from effective Hamiltonian, namely a list with the first term being E0 and next terms being J coefficients
        self.eff_param = eff_param
        self.n = len(eff_param) - 1
        
    def activate_prob(self, point, neigh, J):
        return 1 - np.exp(- 2 * self.beta * J * \ ###check this line!!!
                                  self.spins[point[0],point[1]] * self.spins[neigh[0],neigh[1]])
    
    def add_site(self, point, neigh):
        ### add sites to the cluster (1st NN)
        
        if neigh not in self.cluster:
            prob = self.activate_prob(point, neigh, self.eff_param[1])
            # check if the link is activated
            if rnd.random() < prob:
                self.cluster.append(neigh) # add point to the cluster
                self.extend_cluster(neigh) # extend the cluster from the center positioned at 'neighbour point'
            
    def extend_cluster(self, point):
        ## check if the neighbors should be added to the cluster, in anti-clockwise direction
        self.add_site(point, [(point[0]+1)%self.size, point[1]])
        self.add_site(point, [point[0], (point[1]+1)%self.size])
        self.add_site(point, [(point[0]-1)%self.size, point[1]])
        self.add_site(point, [point[0], (point[1]-1)%self.size])

    def SLMC_Update(self):
        ### 
        
        if self.n == 1:
            E_a = Hamiltonian(self.J, self.K, self.spins)
            E_a_eff = Hamiltonian_eff_1(self.eff_param[0], self.eff_param[1], self.spins) ### note the definition of hamiltonian_eff !!!
        
            #randomly pick a site and add to the cluster
            i, j = rnd.randint(L, size=(2)) 
            self.cluster.append([i,j])
        
            #check adjacent states to build the whole cluster
            self.add_site([i,j],[(i+1) % self.size,j])
            self.add_site([i,j],[i,(j+1) % self.size])
            self.add_site([i,j],[(i-1) % self.size,j])
            self.add_site([i,j],[i,(j-1) % self.size])
        
            #  
            for site in self.cluster:
                self.spins[site[0],site[1]] *= -1 
        
            # self.config.spins = spins
            E_b = Hamiltonian(self.J, self.K, self.spins)
            E_b_eff = Hamiltonian_eff_1(self.eff_param[0], self.eff_param[1], self.spins) 
            energy_diff = (E_b - E_b_eff) - (E_a - E_a_eff)
        
            prob = np.min([1, np.exp(- self.beta * energy_diff)])
        
            # check if we keep the flip
            if rnd.random() < prob:
                return self.spins
            
            else: 
                for site in self.cluster:
                    self.spins[site[0],site[1]] *= -1
                return self.spins