import emcee
from Ising import *
import numpy as np
import matplotlib.pyplot as plt


def emcee_posterior(state, J, h, beta = 1.0):
    state = list_to_grid(np.sign(state))
    E = ising_energy(state, J, h)
    return -beta*E
    
def MHMove_proposal(chains, rng):
    N_chains, N_spins = chains.shape
    spin_swaps = rng.choice(np.arange(N_spins), size = (1,N_chains), replace = True)
    chain_indices = np.reshape(np.arange(N_chains), (1,N_chains))
    proposal = np.copy(chains)
    proposal[chain_indices, spin_swaps]*=-1
    return proposal, 0

def run_emcee(N_walkers, N_grid, J, h, beta = 1.0, seed = 123, steps = 20000):
    rng = np.random.default_rng(seed)
    
    N_spins = N_grid**2
    initial_state = rng.uniform(low = -.01, high = .01, size = (N_walkers, N_spins))
    
    sampler = emcee.EnsembleSampler(N_walkers, N_spins, emcee_posterior, args = [J,h], kwargs = {"beta" : beta}, moves = emcee.moves.MHMove(MHMove_proposal))
    sampler.run_mcmc(initial_state, steps, skip_initial_state_check = True)
    return sampler

if __name__ == "__main__":
    N_walkers = 20
    N_grid = 10
    burn_in_steps = 5000
    chain_steps = 100000
    mcmc = run_emcee(N_walkers, N_grid, 1, .1, seed = None, steps = burn_in_steps + chain_steps)
    chain_array = np.sign(mcmc.get_chain()[burn_in_steps:,:,:])
    tau_emcee = emcee.autocorr.integrated_time(chain_array, quiet = False)
    print(np.max(tau_emcee))
    print(tau_emcee)

    #print(f"ACL: {tau_direct_indep:.2f}")
    #plt.figure()
    #plt.hist(chain_list)
    #plt.show()
    
    chain_array = np.sign(chain_array)
    #print(np.max(chain_array))
    #print(np.min(chain_array))
