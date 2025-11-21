import emcee
from Ising import *
import numpy as np
import matplotlib.pyplot as plt


def emcee_posterior(state, J, h, beta = 1.0):
    state = np.sign(state)
    E = ising_energy(state, J, h)
    return -beta*E
    
def MHMove_proposal(rng, chains):
    N_chains, N_spins = chains.shape
    spin_swaps = rng.choice(np.arange(N_spins), size = (1,N_chains), replace = True)
    chain_indices = np.reshape(np.arange(N_chains), (1,N_chains))
    proposal = np.copy(chains)
    proposal[chain_indices, spin_swaps]*=-1
    return proposal, 0

def run_emcee(N_walkers, N_grid, J, h, beta = 1.0, seed = 123):
    rng = np.random.default_rng(seed)
    
    N_spins = N_grid**2
    initial_state = rng.uniform(low = -10, high = 10, size = (N_walkers, N_spins))
    
    sampler = emcee.EnsembleSampler(N_walkers, N_spins, emcee_posterior, args = [J,h], kwargs = {"beta" : beta}, moves = emcee.moves.MHMove(MHMove_proposal))
    sampler.run_mcmc(initial_state, 20000)
    return sampler

if __name__ == "__main__":
    N_walkers = 3
    N_grid = 5
    mcmc = run_emcee(N_walkers, N_grid, 1, 0, seed = None)
    chain_array = mcmc.get_chain()
    
