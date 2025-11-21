""" This module is to verify convergence in the mcmc chain."""

from Ising import magnetization
import mcmc
import numpy as np
import matplotlib.pyplot as plt

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