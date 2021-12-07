import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import itertools
import matplotlib.animation as animation

class Configuration:
    """A configuration of Ising spins"""

    def __init__(self, T, J, L):
        self.size = L
        self.J = J
        self.beta = 1. / T
        self.spins = rnd.choice([-1, 1], size=(L, L))  # either +1 or -1
        self.energy = self._get_energy()
        self.magnetization = self._get_magnetization()
        self.group = np.zeros((L, L))

    def _get_energy(self):
        """Returns the total energy"""
        energ = 0.
        for i, j in itertools.product(range(self.size), repeat=2):
            energ += -self.J * self.spins[i, j] * (
                        self.spins[i, (j + 1) % self.size] + self.spins[(i + 1) % self.size, j])
        return energ

    def _get_magnetization(self):
        """Returns the total magnetization"""
        magnet = np.sum(self.spins)
        return magnet

    def add_point(self, point, neigh):
        L = self.size
        J = self.J
        spins = self.spins
        beta = self.beta
        re = 0
        if self.group[neigh[0], neigh[1]]!=1:
            p = 1 - np.exp(2*beta*J * spins[point[0], point[1]] * spins[neigh[0], neigh[1]])
            if rnd.random() < p:
                self.spins[neigh[0], neigh[1]] *= -1
                self.magnetization += 2 * self.spins[neigh[0], neigh[1]]
                self.group[neigh[0], neigh[1]] = 1
                re = 1
        return re

    def group_extand(self, i, j):
        L = self.size
        if self.add_point([i, j], [(i-1)%L, j]):
            self.group_extand((i-1)%L, j)
        if self.add_point([i, j], [(i+1)%L, j]):
            self.group_extand((i+1)%L, j)
        if self.add_point([i, j], [i, (j-1)%L]):
            self.group_extand(i, (j-1)%L)
        if self.add_point([i, j], [i, (j+1)%L]):
            self.group_extand(i, (j+1)%L)

def config_to_image(config):
    """Turn an array into an image"""
    L = config.size
    im = np.zeros([L, L, 3])
    for i, j in itertools.product(range(L), repeat=2):
        im[i, j, :] = (1, 0, 0) if config.spins[i, j] == 1 else (0, 0, 0)
    return im

def metropolis_move(config):
    """Modify (or not) a configuration with Metropolis algorithm"""

    L = config.size
    J = config.J
    beta = config.beta
    delta_energy = 0
    config.group = np.zeros((L, L))
    i, j = rnd.randint(L, size=(2))  # pick a random site
    config.group[i, j] = 1
    config.spins[i, j] *= -1
    config.group_extand(i, j)
    for i in range(L):
        for j in range(L):
            if config.group[i, j]:
                coef = 2 * J * config.spins[i, j]
                if config.group[(i-1)%L, j]!=1: delta_energy += coef * config.spins[(i-1)%L, j]
                if config.group[(i+1)%L, j]!=1: delta_energy += coef * config.spins[(i+1)%L, j]
                if config.group[i, (j-1)%L]!=1: delta_energy += coef * config.spins[i, (j-1)%L]
                if config.group[i, (j+1)%L]!=1: delta_energy += coef * config.spins[i, (j+1)%L]

    config.energy += delta_energy

L = 50
T = 1.93
length_cycle = 1
nt = 200

# instantiate a configuration
config = Configuration(T, 1.0, L)

# a two-panel figure
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
im = ax1.imshow(config_to_image(config), interpolation='none')
ax2 = fig.add_subplot(122, aspect=100)
line, = ax2.plot([], [])
ax2.set_xlim(0, nt)
ax2.set_ylim(-1, 1)

steps = []
magnet = []


def do_mc_cycle(n):
    m = 0
    for k in range(length_cycle):
        metropolis_move(config)
        m += np.abs(config.magnetization / float(config.size ** 2))
    m /= length_cycle
    im.set_array(config_to_image(config))
    if len(steps) < nt: steps.append(n)
    if len(magnet) < nt:
        magnet.append(m)
    else:
        magnet.insert(nt, m)
        magnet.pop(0)

    line.set_data(steps, magnet)
    return (im, line)


ani = animation.FuncAnimation(fig, do_mc_cycle, interval=1, blit=False)
plt.show()
