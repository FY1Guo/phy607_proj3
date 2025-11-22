""" This module is to verify convergence in the mcmc chain."""

from Ising import magnetization
from mcmc import *
import numpy as np
import matplotlib.pyplot as plt
from Ising import *

from external import *

import emcee as emcee_lib


def gelman_rubin(chains):
    """
    returns R_hat(Gelman-Rubin statistic)"""

    if isinstance(chains, list):
        chains = np.array(chains)

    if chains.ndim == 1:
        raise ValueError("More than 2 chains required for R_hat")
    
    n_chains = chains.shape[0]
    n_samples = chains.shape[1]

    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)

    B = n_samples / (n_chains - 1)*np.sum((chain_means - overall_mean)**2) #this is the between chain var

    #within chain variance
    chain_variances = np.var(chains, axis=1, ddof = 1)

    W = np.mean(chain_variances)

    var_plus = ((n_samples - 1) / n_samples) * W * (1 / n_samples) * B

    R_hat = np.sqrt(var_plus / W)

    return R_hat

def direct_autocorr(chain, max_lag=None):
    # first half is burn-in
    chain = chain[len(chain)//2:]
    
    n = len(chain)
    if max_lag is None:
        max_lag = n // 10  
    
    
    y = chain - np.mean(chain)
    
    
    c = np.correlate(y, y, mode='full')
    c = c[n-1 : n-1+max_lag]  
    rho = c / c[0]  #normalize so rho[0] is 1.0
    

    negative_indices = np.where(rho < 0)[0]
    if len(negative_indices) > 0:
        cutoff = negative_indices[0]
    else:
        cutoff = len(rho)
    
    tau_estimate = 1.0 + 2.0 * np.sum(rho[1:cutoff])
    
    return tau_estimate, rho

def plot_trace(chains_dict, param_names=None, save_path=None, 
               true_values=None, burn_in=None):

    if param_names is None:
        param_names = list(chains_dict.keys())
    
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3 * n_params))
    
    if n_params == 1:
        axes = [axes]
    
    for idx, param in enumerate(param_names):
        chains = chains_dict[param]
        ax = axes[idx]
        
        
        if isinstance(chains, np.ndarray):
            if chains.ndim == 1:
                chains = [chains]
            else:
                chains = [chains[i] for i in range(chains.shape[0])]
        
       
        for i, chain in enumerate(chains):
            iterations = np.arange(len(chain))
            ax.plot(iterations, chain, alpha=0.6, linewidth=0.8, 
                   label=f'Chain {i+1}')
        
       
        if burn_in is not None:
            ax.axvline(burn_in, color='red', linestyle='--', 
                      alpha=0.5, linewidth=2, label='Burn-in')
        
       
        if true_values is not None and param in true_values:
            ax.axhline(true_values[param], color='green', 
                      linestyle='--', linewidth=2, alpha=0.7,
                      label=f'True {param}')
        
        ax.set_ylabel(param, fontsize=12)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_title(f'Trace Plot: {param}', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def mag_trace_plot(chain, burn_frac=0, save_path=None, label=None, title=None):
    """
    Given a chain (list of grids), make a magnetization trace plot.
    """
    burn_in = int(len(chain) * burn_frac)
    grids = chain[burn_in:]
    mags = np.array([magnetization(g) for g in grids])

    plt.figure()
    plt.plot(mags, label=label if label is not None else None)
    plt.xlabel("MC step")
    plt.ylabel("Magnetization per spin")
    if title is not None:
        plt.title(title)
    if label is not None:
        plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    return mags

def spin_up_probability(chains, spin_num = 0):
    """
    Given an ensemble of chains, plot the ratio of spin up to spin down for a given spin over time
    """
    min_length = np.inf
    N_chains = len(chains)
    for chain in chains:
        if len(chain)<min_length:
            min_length = len(chain)
    spin_array = np.ones((N_chains, min_length))
    for i in range(min_length):
        for j, chain in enumerate(chains):
            spin_array[j,i] = chain[i][spin_num]
    plt.figure()
    plt.plot(np.sum(spin_array, axis = 0))
    plt.xlabel("MC step")
    plt.ylabel("Net spin over all chains")
    plt.title(f"Spin number {spin_num}")

N = 10 
beta = 0.3
J = 1.0
h = 1.0
iterations = 20000
burn_frac = 0.3

sampler_emcee = run_emcee(N_walkers=20,
    N_grid=N,
    J=J,
    h=h,
    beta=beta,
    seed=200,
    steps=iterations
)



print("Running 4 chains for Gelman-Rubin test...")
chains_list = []
mag_chains = []

for i in range(4):
    print(f"\nChain {i+1}/4...")
    rng = np.random.default_rng(seed=100+i)
    init_grid = rng.choice([-1, 1], size=(N, N))
    
    chain, _ = run_chain(iterations, init_grid, posterior, 
                         proposal_func_single, J, h, beta)
    
    chains_list.append(chain)
    
    # Extract magnetization
    mags = np.array([magnetization(g) for g in chain])
    mag_chains.append(mags)



# Gelman-Rubin test
mag_chains_array = np.array(mag_chains)
R_hat = gelman_rubin(mag_chains_array)

print(f"\n{'='*60}")
print(f"CONVERGENCE RESULTS")
print(f"{'='*60}")
print(f"Gelman-Rubin R-hat: {R_hat:.4f}")
if R_hat < 1.1:
    print("CONVERGED (R-hat < 1.1)")
else:
    print("NOT CONVERGED (R-hat >= 1.1)")

# Autocorrelation analysis
tau, rho = direct_autocorr(mag_chains[0])
ess = len(mag_chains[0]) / tau

print(f"\nAutocorrelation time (τ): {tau:.2f}")
print(f"Effective Sample Size: {ess:.0f} / {len(mag_chains[0])}")
print(f"{'='*60}\n")


chain_emcee = np.sign(sampler_emcee.get_chain())
n_steps, n_walkers, n_spins = chain_emcee.shape

print(f"Chain shape: {chain_emcee.shape}")

# Extract magnetization for each walker
mag_chains_emcee = []
for walker_idx in range(n_walkers):
    mags = []
    for step in range(n_steps):
        grid = list_to_grid(chain_emcee[step, walker_idx, :])
        mags.append(magnetization(grid))
    mag_chains_emcee.append(np.array(mags))

mag_chains_emcee = np.array(mag_chains_emcee)

# Gelman-Rubin test for emcee
R_hat_emcee = gelman_rubin(mag_chains_emcee)

print(f"\nGelman-Rubin R-hat (emcee): {R_hat_emcee:.4f}")
if R_hat_emcee < 1.1:
    print("CONVERGED (R-hat < 1.1)")
else:
    print("NOT CONVERGED (R-hat >= 1.1)")

# Autocorrelation analysis for emcee (using first walker)

tau_emcee, rho_emcee = direct_autocorr(mag_chains_emcee[0])
ess_emcee = len(mag_chains_emcee[0]) / tau_emcee

print(f"\nAutocorrelation time (τ): {tau_emcee:.2f}")
print(f"Effective Sample Size: {ess_emcee:.0f} / {len(mag_chains_emcee[0])}")

# emcee's built-in autocorrelation
try:
    tau_emcee_builtin = emcee_lib.autocorr.integrated_time(chain_emcee, quiet=True)
    print(f"emcee built-in τ (avg): {np.mean(tau_emcee_builtin):.2f}")
    print(f"emcee built-in τ (range): [{np.min(tau_emcee_builtin):.2f}, {np.max(tau_emcee_builtin):.2f}]")
except Exception as e:
    print(f"emcee built-in autocorr failed: {e}")

"""
# Plot traces
chains_dict = {'Magnetization': mag_chains}
fig = plot_trace(chains_dict, burn_in=int(iterations*burn_frac))
plt.savefig('convergence_traces.png', dpi=300, bbox_inches='tight')
print("Saved: convergence_traces.png")


# Plot ACF
plt.figure(figsize=(10, 5))
lags = np.arange(len(rho))
plt.plot(lags, rho, linewidth=2)
plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
plt.axhline(0.05, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Autocorrelation', fontsize=12)
plt.title(f'Magnetization ACF (τ={tau:.1f})', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('acf_plot.png', dpi=300, bbox_inches='tight')
print("Saved: acf_plot.png")
"""
print("\nCreating comparison plots")

# side-by-side trace plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# metropolis traces
ax = axes[0]
for i in range(4):
    ax.plot(mag_chains[i], alpha=0.6, linewidth=0.8, label=f'Chain {i+1}')
ax.axvline(int(iterations*burn_frac), color='red', linestyle='--', 
           linewidth=2, label='Burn-in')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Magnetization per spin', fontsize=12)
ax.set_title('Metropolis (Hand-written)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# emcee traces
ax = axes[1]
for i in range(4):  # Show first 4 walkers
    ax.plot(mag_chains_emcee[i], alpha=0.6, linewidth=0.8, label=f'Walker {i+1}')
ax.axvline(int(iterations*burn_frac), color='red', linestyle='--', 
           linewidth=2, label='Burn-in')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Magnetization per spin', fontsize=12)
ax.set_title('emcee', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('trace_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: trace_comparison.png")

# 2. side-by-side ACF plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# metropolis ACF
ax = axes[0]
lags = np.arange(len(rho))
ax.plot(lags, rho, linewidth=2, color='blue')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(0.05, color='red', linestyle='--', alpha=0.3)
ax.axhline(-0.05, color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Autocorrelation', fontsize=12)
ax.set_title(f'Metropolis ACF (τ={tau:.1f})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.2, 1.1])

# emcee ACF
ax = axes[1]
lags_emcee = np.arange(len(rho_emcee))
ax.plot(lags_emcee, rho_emcee, linewidth=2, color='orange')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(0.05, color='red', linestyle='--', alpha=0.3)
ax.axhline(-0.05, color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Autocorrelation', fontsize=12)
ax.set_title(f'emcee ACF (τ={tau_emcee:.1f})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.2, 1.1])

plt.tight_layout()
plt.savefig('acf_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: acf_comparison.png")


plt.show()