""" This module is to verify convergence in the mcmc chain."""

import Ising
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

