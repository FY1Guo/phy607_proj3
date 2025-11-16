import numpy as np
import tdqm

from Ising import ising_energy, grid_to_list, list_to_grid

def posterior(state, J, h) #Maybe takes other parameters?)
    return

def run_chain(iterations, initial_cond, posterior, proposal_func, J, h):
    chain = []
    probabilities = []
    
    state = np.copy(initial_cond)
    initial_prob = posterior(initial_cond)
    probabilities.append(initial_prob)
    
    for i in tdqm.tdqm(range(iterations)):
        state_test = proposal_func(state[-1])#, J, h)
        p_test = posterior(state_test)
        u = np.random.uniform(0,1)
        acceptance_prob = p_test - p_list[-1]
        if np.log(u) <= acceptance_prob:
            chain.append(state_test)
            p_list.append(p_test)
    return chain_list, p_list
