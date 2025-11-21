import numpy as np
from Ising import *
import matplotlib.pyplot as plt

N = 4
N_spins = N**2
N_states = 2**N_spins

states = []

for i in range(N_states):
    rep = [int(digit) for digit in bin(i)[2:]]
    if len(rep)<N_spins:
        for i in range(N_spins-len(rep)):
            rep.insert(0,0)
    rep = np.array(rep)
    rep = rep*2-1
    #print(rep)
    rep = list_to_grid(rep)
    states.append(rep)
    
beta = .1
J = 1
h = .5

Z = 0
probs = []
mags = []
for state in states:
    p = np.exp(-1*ising_energy(state, J, h)*beta)
    Z += p
    probs.append(p)
    mags.append(magnetization(state))

mag_histogram_mags = []
mag_histogram_probs = []
for i in range(N_states):
    if mags[i] in mag_histogram_mags:
        idx = mag_histogram_mags.index(mags[i])
        mag_histogram_probs[idx] += probs[i]/Z
    else:
        mag_histogram_mags.append(mags[i])
        mag_histogram_probs.append(probs[i]/Z)
plt.hist(mag_histogram_mags, weights = mag_histogram_probs)

plt.show()
