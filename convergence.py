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

