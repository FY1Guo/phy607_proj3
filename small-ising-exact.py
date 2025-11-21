import numpy as np 

from Ising import * 

import matplotlib.pyplot as plt 

def exact_distribution(N, beta, J, h):

    N_spins = N**2
    
    N_states = 2**N_spins
    
    print(f"Computing exact solution for {N}×{N} grid ({N_states} states)...")
    
    states = []
    energies = []
    mags = []
    
    #all possible configurations
    for i in range(N_states):
        # Convert integer to binary representation
        rep = [int(digit) for digit in bin(i)[2:]]
        
        #pad with zeros to get full length
        if len(rep) < N_spins:
            rep = [0] * (N_spins - len(rep)) + rep
        
        #convert 0,1 to -1,+1
        rep = np.array(rep) * 2 - 1
        
        #reshape to grid
        grid = list_to_grid(rep)
        
        #compute observables
        E = ising_energy(grid, J, h)
        m = magnetization(grid)
        
        states.append(grid)
        energies.append(E)
        mags.append(m)
    
    states = np.array(states)
    energies = np.array(energies)
    mags = np.array(mags)
    
    #compute Boltzmann weights
    boltzmann_weights = np.exp(-beta * energies)
    Z = np.sum(boltzmann_weights)
    probs = boltzmann_weights / Z
    
    #group by magnetization value
    unique_mags = np.unique(mags)
    mag_probs = np.zeros_like(unique_mags)
    
    for i, m_val in enumerate(unique_mags):
        mask = (mags == m_val)
        mag_probs[i] = np.sum(probs[mask])
    
    print(f"Partition function Z = {Z:.6e}")
    print(f"Number of unique magnetization values: {len(unique_mags)}")
    
    return unique_mags, mag_probs, Z, states, probs

def exact_observables(N, beta, J, h):
    
    N_spins = N**2
    N_states = 2**N_spins
    
    energies = []
    mags = []
    
    for i in range(N_states):
        rep = [int(digit) for digit in bin(i)[2:]]
        if len(rep) < N_spins:
            rep = [0] * (N_spins - len(rep)) + rep
        rep = np.array(rep) * 2 - 1
        grid = list_to_grid(rep)
        
        E = ising_energy(grid, J, h)
        m = magnetization(grid)
        
        energies.append(E)
        mags.append(m)
    
    energies = np.array(energies)
    mags = np.array(mags)
    
    #partition function
    boltzmann_weights = np.exp(-beta * energies)
    Z = np.sum(boltzmann_weights)
    probs = boltzmann_weights / Z
    
    #expectation values
    mean_energy = np.sum(probs * energies) / N_spins
    mean_mag = np.sum(probs * np.abs(mags))
    mean_mag_sq = np.sum(probs * mags**2)
    mean_energy_sq = np.sum(probs * energies**2)
    
    #fluctuations
    mag_susceptibility = beta * N_spins * (mean_mag_sq - mean_mag**2)
    heat_capacity = (beta**2 / N_spins) * (mean_energy_sq - (np.sum(probs * energies))**2)
    
    results = {
        'mean_mag': mean_mag,
        'mean_energy': mean_energy,
        'mag_susceptibility': mag_susceptibility,
        'heat_capacity': heat_capacity,
        'Z': Z
    }
    
    return results



