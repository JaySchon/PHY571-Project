# This file implement Wolff update algorithm for the system with original Hamiltonian

### import libraries
import numpy as np
import numpy.random as rnd
from Hamiltonian import four_body_sum

class WolffUpdate():
    """A class to implememnt Wolff update method."""
    def __init__(self, spins, J, K, T):
        """Initiate parameters"""
        self.spins = spins
        self.size = len(spins)
        self.cluster = []
        self.J = J
        self.K = K
        self.T = T
        self.beta = 1./ T
        
    def activate_prob(self, point, neigh):
        """Calculate the actiavation probability of the link."""
        return 1 - np.exp(- 2 * self.beta * self.J * \
                                  self.spins[point[0],point[1]] * self.spins[neigh[0],neigh[1]])
    
    def add_site(self, point, neigh):
        """Add sites to the cluster (1st NN)"""
        
        if neigh not in self.cluster:
            prob = self.activate_prob(point, neigh)
            # check if the link is activated
            if rnd.random() < prob:
                self.cluster.append(neigh) # add point to the cluster
            
                self.extend_cluster(neigh) # extend the cluster from the center positioned at 'neighbour point'
            
    def extend_cluster(self, point):
        """To extend the cluster by checking all the nearest neighbours."""
        size = len(self.spins)
        ## check if the neighbors should be added to the cluster, in anti-clockwise direction
        self.add_site(point, [(point[0]+1)%size, point[1]])
        self.add_site(point, [point[0], (point[1]+1)%size])
        self.add_site(point, [(point[0]-1)%size, point[1]])
        self.add_site(point, [point[0], (point[1]-1)%size])
        
    
    def Wolff_Update_0(self):
        """
        To implement Wolff Update for the System with Hamiltonian: H = -J \sum_{<i,j>}S_i*S_j
        Output:
        the spins configuration matrix after one Wolff update.
        """
        
        ### Build the cluster and flip it with probability one 
        
        #randomly pick a site and add to the cluster
        i, j = rnd.randint(self.size, size=(2)) 
        self.cluster.append([i,j])
        
        #check adjacent states to build the whole cluster
        self.add_site([i,j],[(i+1) % self.size,j])
        self.add_site([i,j],[i,(j+1) % self.size])
        self.add_site([i,j],[(i-1) % self.size,j])
        self.add_site([i,j],[i,(j-1) % self.size])
        
        # flip all the site in the cluster with prob 1
        for site in self.cluster:
            self.spins[site[0],site[1]] *= -1 
        return self.spins
        
        
    def Wolff_Update_1(self):
        """
        To implement Wolff Update for the System with original Hamiltonian.
        Output:
        the spins configuration matrix after one Wolff update.
        """
        
        ### build the cluster and flip it with probability relevant to external sources 
       
        energy_1_ini = -self.K * four_body_sum(self.spins)
        
        #randomly pick a site and add to the cluster
        i, j = rnd.randint(self.size, size=(2)) 
        self.cluster.append([i,j])
        
        #check adjacent states to build the whole cluster
        self.add_site([i,j],[(i+1) % self.size,j])
        self.add_site([i,j],[i,(j+1) % self.size])
        self.add_site([i,j],[(i-1) % self.size,j])
        self.add_site([i,j],[i,(j-1) % self.size])
        
        # flip the cluster and calclate the energy difference of the external source term
        for site in self.cluster:
            self.spins[site[0],site[1]] *= -1 
        
        # self.config.spins = spins
        energy_1_flip = -self.K * four_body_sum(self.spins)
        energy_1_diff = energy_1_flip - energy_1_ini
        
        prob = np.min([1, np.exp(- self.beta * energy_1_diff)])
        
        # check if we keep the flip
        if rnd.random() < prob:
            return self.spins
            
        else: 
            for site in self.cluster:
                self.spins[site[0],site[1]] *= -1
            return self.spins

