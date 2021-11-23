import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from Configuration import Configuration
from WolffUpdate import WolffUpdate
from LocalUpdate import LocalUpdate
from SelfLearningUpdate import SelfLearningUpdate
import matplotlib.animation as animation
import itertools

L = 25
J = 1.
K = 0.2
T = 2.1
n_cycles = 1000    # cycles shown
type = 0     # local update type = 0; global update type =1; SLMC update type = 2
Ncycle = L**2     # local updates in one cycle
eff_param = [-100, 1.1]     # Heff parameters


def config_to_image(spins):
    """Turn a spins array into an image"""
    L = len(spins)
    im = np.zeros([L,L,3])
    for i,j in itertools.product(range(L), repeat=2):
        im[i,j,:] = (1,0,0) if spins[i,j]==1 else (0,0,0)
    return im

spins = rnd.choice([-1,1],size = (L, L))
config = Configuration(spins, L, J, K, T)

# a two-panel figure
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(221)
im = ax1.imshow(config_to_image(config.spins), interpolation='none')

ax2 = fig.add_subplot(222)
line2, = ax2.plot([], [])
ax2.set_title("Magnetization", fontsize=25)
ax2.set_xlabel("$t$", fontsize=20)
ax2.set_ylabel("|m|", fontsize=20)
ax2.set_xlim(0, n_cycles)
ax2.set_ylim(0, 1)

ax3 = fig.add_subplot(223)
line3, = ax3.plot([], [])
ax3.set_title("Average Energy", fontsize=25)
ax3.set_xlabel("$t$", fontsize=20)
ax3.set_ylabel("E/N", fontsize=20)
ax3.set_xlim(0, n_cycles)
ax3.set_ylim(-2*J-K, 0)

steps = []
mr = []
e = []

def do_mc_cycle(n):
    global config
    if type == 1:
        update = WolffUpdate(config.spins, J, K, T)
        spins = update.Wolff_Update_1()
    if type == 0:
        update = LocalUpdate(config.spins, J, K, T)
        for i in range(Ncycle-1): update.local_update()
        spins = update.local_update()
    if type == 2:
        update = SelfLearningUpdate(config.spins, J, K, T, eff_param)
        spins = update.SLMC_Update()
    config = Configuration(spins, L, J, K, T)
    steps.append(n)
    mr.append(abs(config.magnetization)/L**2)
    e.append(config.energy/L**2)
    line2.set_data(steps, mr)
    line3.set_data(steps, e)
    im.set_array(config_to_image(config.spins))
    return (im, line2, line3)


ani = animation.FuncAnimation(fig, do_mc_cycle, interval=1, blit=False)
plt.show()