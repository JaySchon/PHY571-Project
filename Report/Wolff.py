### build a class for the system
import numpy as np
import numpy.random as rnd
import itertools
import matplotlib.pyplot as plt



class Configuration:
    """A configuration of Ising spins."""
    
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
        

### build a class for the system
import numpy as np
import numpy.random as rnd
import itertools
import matplotlib.pyplot as plt



class Configuration0:
    """A configuration of Ising spins."""
    
    def __init__(self, L, J, T):
        self.size = L
        self.J = J
        self.beta = 1./ ( T)
        self.spins = rnd.choice([-1,1],size = (L, L))  #create a random spin configuration
        self.energy = self._get_energy()
        self.magnetization = self._get_magnetization()
        
    def _get_energy(self):
        """Return the total energy"""
        ### Define a function to calculate the energy of the system, given the Hamiltonian: 
        ### H = -J sum_{<ij>} S_iS_j - K sum_{<ijkl>} S_i S_j S_k S_l
        energ = 0
        ### calculate the two-body sum
        for i,j in itertools.product(range(self.size), repeat=2):
            energ += -self.J * self.spins[i,j] * (self.spins[i,(j+1)%self.size] + self.spins[(i+1)%self.size,j]) # only consider the right and top part for each site
        
        ### calculate the four-body sum
        #for i,j in itertools.product(range(self.size), repeat=2):
        #    energ += -self.K * self.spins[i,j] * self.spins[i,(j+1)%self.size] * self.spins[(i+1)%self.size,j] * self.spins[(i+1)%self.size, (j+1)%self.size] # only consider the right and top part for each site
        
        return energ
    
    def _get_magnetization(self):
        """Return the total magnetization"""
        magnet = (np.sum(self.spins))
        return magnet 
    
    
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
        ### Build the cluster and flip it with probability one 
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



        
L = 10
T = 4
config = Configuration(L, 1.0, 0.2, T)
print(config.spins)
print(config.energy)
print(config.magnetization)
update = WolffUpdator(config)
print(update.cluster)
config = update.Wolff_Update_0()
print(update.cluster)
print(config.spins)
print(config.energy)
print(config.magnetization)
