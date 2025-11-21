import numpy as np
import tqdm
import pickle
import os
from multiprocessing import Pool

from Ising import ising_energy, grid_to_list, list_to_grid


def posterior(state, J, h, beta=1.0):  # Maybe takes other parameters?)
    E = ising_energy(state, J, h)
    return -beta * E


def proposal_func_single(state):
    """
    Propose a new state by flipping a random spin in the grid.
    """
    grid = np.copy(state)
    N = grid.shape[0]
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    grid[i, j] *= -1
    return grid


def proposal_func_single_fractional(state, frac=0.1):
    """
    Propose a new state by flipping fraction of spins in the grid.
    """
    grid = np.copy(state)
    array = grid_to_list(grid)
    N = len(array)
    n_flip = int(frac * N)
    flipped_indices = np.random.choice(N, n_flip, replace=False)
    array[flipped_indices] *= -1
    return list_to_grid(array)


def run_chain(iterations, initial_cond, posterior, proposal_func, J, h, beta=1.0):
    chain = []
    log_probs = []

    state = np.copy(initial_cond)
    current_prob = posterior(state, J, h, beta)
    chain.append(np.copy(state))
    log_probs.append(current_prob)

    for i in tqdm.tqdm(range(iterations)):
        state_test = proposal_func(state)  # , J, h)
        p_test = posterior(state_test, J, h, beta)
        u = np.random.uniform(0, 1)
        acceptance_prob = p_test - current_prob
        if np.log(u) <= acceptance_prob:
            state = state_test
            current_prob = p_test

        chain.append(np.copy(state))
        log_probs.append(current_prob)

    return chain, log_probs


def run_and_save_chain(
    iterations, initial_cond, posterior, proposal_func, J, h, beta=1.0, chain_num=0
):
    chain, log_probs = run_chain(
        iterations, initial_cond, posterior, proposal_func, J, h, beta=1.0, chain_num=0
    )
    chain_label = f"chain{chain_num}"
    with open(chain_label, "wb") as f:
        pickle.dump(chain, f)


def ensemble_sampler(
    N_chains,
    iterations,
    initial_conds,
    posterior,
    proposal_func,
    J,
    h,
    beta=1.0,
    threads=1,
):
    sample_chain = lambda x: run_and_save_chain(
        iterations,
        initial_conds[x],
        posterior,
        proposal_func,
        J,
        h,
        beta=beta,
        chain_num=x,
    )
    with Pool(threads) as p:
        p.map(sample_chain, np.arange(N_chains))
    chain_list = []
    for i in range(N_chains):
        with open(f"chain{i}", "rb") as f:
            chain = pickle.load(f)
            chain_list.append(chain)
        if os.path.exists(f"chain{i}"):
            os.remove(f"chain[i]")
    with open("Compiled-Chains", "rb") as f:
        pickle.dump(chain_list, f)
