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
    
    