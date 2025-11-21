"""

Compare MCMC with results to exact solution. 

"""

import numpy as np 

import matplotlib.pyplot as plt

from smallisingexact import * 

from mcmc import *
from Ising import *
from convergence import *

def validate_small_grid(N=3, beta = 1.0, J = 1.0, h=0.0, 
                        iterations = 50000, burn_frac=0.3, seed = 42):

  
    print(f"MCMC VALIDATION: {N}×{N} Ising Model")
   
    print(f"Parameters: β={beta:.3f}, J={J:.3f}, h={h:.3f}")
    print(f"MCMC iterations: {iterations}, burn-in: {burn_frac*100:.0f}%")
    print()
    
    # Step 1: Compute exact solution
    print("Computing exact solution...")
    exact_mags, exact_probs, Z, _, _ = exact_distribution(N, beta, J, h)
    exact_obs = exact_observables(N, beta, J, h)
    
    print(f"\nExact Results:")
    print(f"  <|m|> = {exact_obs['mean_mag']:.6f}")
    print(f"  <E>/N = {exact_obs['mean_energy']:.6f}")
    print(f"  Z = {exact_obs['Z']:.6e}")
    print()
    
    # Step 2: Run MCMC
    print("Running MCMC...")
    rng = np.random.default_rng(seed)
    init_grid = rng.choice([-1, 1], size=(N, N))
    
    chain, log_probs = run_chain(
        iterations, init_grid, posterior, proposal_func_single, J, h, beta
    )
    
    # Step 3: Extract observables from MCMC
    burn_in = int(iterations * burn_frac)
    chain_postburn = chain[burn_in:]
    
    mcmc_mags = np.array([magnetization(g) for g in chain_postburn])
    mcmc_energies = np.array([energy_per_spin(g, J, h) for g in chain_postburn])
    
    mcmc_mean_mag = np.mean(np.abs(mcmc_mags))
    mcmc_mean_energy = np.mean(mcmc_energies)
    mcmc_std_mag = np.std(mcmc_mags)
    mcmc_std_energy = np.std(mcmc_energies)
    
    print(f"\nMCMC Results ({len(mcmc_mags)} samples post-burn-in):")
    print(f"  <|m|> = {mcmc_mean_mag:.6f} ± {mcmc_std_mag/np.sqrt(len(mcmc_mags)):.6f}")
    print(f"  <E>/N = {mcmc_mean_energy:.6f} ± {mcmc_std_energy/np.sqrt(len(mcmc_mags)):.6f}")
    print()

    #STEP HERE FOR COMPARISON MAYBE

    #STEP 3: CONVERGENCE

        # Step 5: Convergence diagnostics
    print("Computing convergence diagnostics...")
    
    # Full chain for autocorrelation (includes burn-in handling internally)
    all_mags = np.array([magnetization(g) for g in chain])
    tau, rho = direct_autocorr(all_mags)
    ess = len(all_mags) / tau
    
    print(f"  Autocorrelation time: τ = {tau:.2f}")
    print(f"  Effective sample size: {ess:.0f} / {len(all_mags)}")
    print()
    