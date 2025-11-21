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

def plot_exact_vs_mcmc(exact_mags, exact_probs, mcmc_mags, 
                       beta, J, h, save_path='comparison.png'):

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    #exact distribution (bar plot)
    ax1.bar(exact_mags, exact_probs, width=0.05, alpha=0.7, 
            color='blue', edgecolor='black', label='Exact')
    ax1.set_xlabel('Magnetization per spin', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title(f'Exact Distribution\nβ={beta:.2f}, J={J:.2f}, h={h:.2f}', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    #NCMC histogram
    ax2.hist(mcmc_mags, bins=30, density=True, alpha=0.6, 
             color='red', edgecolor='black', label='MCMC')
    #exact as line
    ax2.plot(exact_mags, exact_probs, 'bo-', linewidth=2, 
             markersize=8, label='Exact', alpha=0.8)
    ax2.set_xlabel('Magnetization per spin', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title(f'MCMC vs Exact\n{len(mcmc_mags)} samples', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    
    return fig

