### Wolff update algorithm
### import libraries
import numpy as np
import numpy.random as rnd


class WolffUpdator():
    
    def __init__(self, config):
        self.config = config
        self.cluster = []
        self.J = config.J
        
        
    def activate_prob(self, point, neigh):
        return 1 - np.exp(- 2 * self.config.beta * self.J * \
                                  self.config.spins[point[0],point[1]] * self.config.spins[neigh[0],neigh[1]])
    
    def add_site(self, point, neigh):
        ### add sites to the cluster (1st NN)
        size = self.config.size
        
        if neigh not in self.cluster:
            prob = self.activate_prob(point, neigh)
            # check if the link is activated
            if rnd.random() < prob:
                # self.config.spins(neigh) *= -1
                self.cluster.append(neigh)
            
                self.extend_cluster(neigh)
            
    def extend_cluster(self, point):
        size = self.config.size
        ## check if the neighbors should be added to the cluster, in anti-clockwise direction
        self.add_site(point, [(point[0]+1)%size, point[1]])
        self.add_site(point, [point[0], (point[1]+1)%size])
        self.add_site(point, [(point[0]-1)%size, point[1]])
        self.add_site(point, [point[0], (point[1]-1)%size])
        
    
    def Wolff_Update_0(self):
        ### Build the cluster and flip it with probability one (no external source of energy)
        L = self.config.size
        
        #randomly pick a site and add to the cluster
        i, j = rnd.randint(L, size=(2)) 
        self.cluster.append([i,j])
        
        #check adjacent states to build the whole cluster
        self.add_site([i,j],[(i+1) % L,j])
        self.add_site([i,j],[i,(j+1) % L])
        self.add_site([i,j],[(i-1) % L,j])
        self.add_site([i,j],[i,(j-1) % L])
        
        
        # flip all the site in the cluster with prob 1
        for site in self.cluster:
            self.config.spins[site[0],site[1]] *= -1 
        
        self.config.energy = self.config._get_energy()
        self.config.magnetization = self.config._get_magnetization()
        
        return self.config
        
        
    def Wolff_Update_1(self):
        ### build the cluster and flip it with probability relevant to external sources 
        L = self.config.size
        
        energy_1_ini = self.config.get_energy_1()
        
        
        #randomly pick a site and add to the cluster
        i, j = rnd.randint(L, size=(2)) 
        self.cluster.append([i,j])
        
        #check adjacent states to build the whole cluster
        self.add_site([i,j],[(i+1) % L,j])
        self.add_site([i,j],[i,(j+1) % L])
        self.add_site([i,j],[(i-1) % L,j])
        self.add_site([i,j],[i,(j-1) % L])
        
        # flip the cluster and calclate the energy difference of the external source term
        for site in self.cluster:
            self.config.spins[site[0],site[1]] *= -1 
        
        # self.config.spins = spins
        energy_1_flip = self.config.get_energy_1()
        energy_1_diff = energy_1_flip - energy_1_ini
        
        prob = np.min([1, np.exp(- self.config.beta * energy_1_diff)])
        
        # check if we keep the flip
        if rnd.random() < prob:
            self.config.energy = self.config._get_energy()
            self.config.magnetization = self.config._get_magnetization()
            return self.config
            
        else: 
            for site in self.cluster:
                self.config.spins[site[0],site[1]] *= -1
            self.config.energy = self.config._get_energy()
            self.config.magnetization = self.config._get_magnetization()
            return self.config