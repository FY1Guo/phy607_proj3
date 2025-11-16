import numpy as np
import tqdm

from Ising import ising_energy, grid_to_list, list_to_grid


def posterior(state, J, h, beta=1.0): #Maybe takes other parameters?)
    E = ising_energy(state, J, h)
    return -beta * E


def proposal_func(state): 
    """
    Propose a new state by flipping a random spin in the grid.
    """
    grid = np.copy(state)
    N = grid.shape[0]
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    grid[i, j] *= -1
    return grid


def run_chain(iterations, initial_cond, posterior, proposal_func, J, h, beta=1.0):
    chain = []
    log_probs = []
    
    state = np.copy(initial_cond)
    current_prob = posterior(state, J, h, beta)
    chain.append(np.copy(state))
    log_probs.append(current_prob)
    
    for i in tqdm.tqdm(range(iterations)):
        state_test = proposal_func(state)#, J, h)
        p_test = posterior(state_test, J, h, beta)
        u = np.random.uniform(0, 1)
        acceptance_prob = p_test - current_prob
        if np.log(u) <= acceptance_prob:
            state = state_test
            current_prob = p_test

        chain.append(np.copy(state))
        log_probs.append(current_prob)

    return chain, log_probs
