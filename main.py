import numpy as np
import matplotlib.pyplot as plt

from mcmc import *
from Ising import magnetization, energy_per_spin
from external import run_emcee


def plot_grid(grid, title, fname):
    plt.figure(figsize=(4, 4))
    im = plt.imshow(grid)
    cbar = plt.colorbar(im, ticks=[-1, 1])
    cbar.set_label("spin")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


def run_simulation(N, beta=1.0, J=1.0, h=0.0, iterations=10000, emcee_walkers=20, burn_frac=0.5, seed=123):
    rng = np.random.default_rng(seed)
    init_grid = rng.choice([-1, 1], size=(N, N))

    plot_grid(init_grid, f"Initial configuration (N={N})", "grid_initial.png")

    # ---------- MCMC ----------
    #chain_m, logp_m = run_chain(iterations, init_grid, posterior, lambda x: proposal_func_single_fractional(), J, h, beta)
    chain_m, logp_m = run_chain(iterations, init_grid, posterior, proposal_func_single, J, h, beta)

    burn_m = int(iterations * burn_frac)
    grids_m = chain_m[burn_m:]
    mags_m = np.array([magnetization(g) for g in grids_m])
    energies_m = np.array([energy_per_spin(g, J, h) for g in grids_m])

    plot_grid(grids_m[-1], f"Final configuration (MCMC, N={N})", "grid_final_mcmc.png")

    # ---------- emcee ----------
    sampler = run_emcee(
        N_walkers=emcee_walkers,
        N_grid=N,
        J=J,
        h=h,
        beta=beta,
        seed=seed,
        steps=iterations,
    )

    chain_e = np.sign(sampler.get_chain()[burn_m:, :, :])
    steps_post, n_walkers, n_spins = chain_e.shape

    # For a time trace, just pick one walker (e.g. walker 0)
    states_w0 = chain_e[:, 0, :]

    mags_e = []
    energies_e = []
    for s in states_w0:
        grid = list_to_grid(s)
        mags_e.append(magnetization(grid))
        energies_e.append(energy_per_spin(grid, J, h))
    mags_e = np.array(mags_e)
    energies_e = np.array(energies_e)

    plt.figure()
    plt.plot(mags_m, label="Metropolis")
    plt.plot(mags_e, label="emcee")
    plt.xlabel("MC step (post burn-in)")
    plt.ylabel("Magnetization per spin")
    plt.title("Magnetization trace: Metropolis vs emcee")
    plt.legend()
    plt.tight_layout()
    plt.savefig("magnetization_trace_comp.png", dpi=200, bbox_inches="tight")
    #plt.close()

    plt.figure()
    plt.plot(energies_m, label="Metropolis")
    plt.plot(energies_e, label="emcee")
    plt.xlabel("MC step (post burn-in)")
    plt.ylabel("Energy per spin")
    plt.title("Energy trace: Metropolis vs emcee")
    plt.legend()
    plt.tight_layout()
    plt.savefig("energy_trace_comp.png", dpi=200, bbox_inches="tight")
    #plt.close()
    
    plt.show()


def plot_spin_snapshots(beta_list, grid_list, N, filename="ising_snapshots.png"):
    """Plot one snapshot per beta in a grid layout."""

    n = len(beta_list)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))

    axes = axes.flatten()

    for i, (beta, grid) in enumerate(zip(beta_list, grid_list)):
        ax = axes[i]
        ax.imshow(grid, cmap="viridis", interpolation="none")
        ax.set_title(r"$\beta = %.2f$" % beta)
        ax.set_xticks([])
        ax.set_yticks([])

    # Turn off unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()


def temperature_scan(N, beta_min, beta_max, J=1.0, h=0.0, iterations=10000, emcee_walkers=20, seed=456):
    avg_mags_m = []
    avg_energies_m = []

    avg_mags_e = []
    avg_energies_e = []

    rng = np.random.default_rng(seed)
    betas = np.linspace(beta_min, beta_max, 20)

    snapshot_grids = []

    for beta in betas:
        # ---------- MCMC ----------
        init_grid = rng.choice([-1, 1], size=(N, N))

        chain_m, logp_m = run_chain(iterations, init_grid, posterior, proposal_func_single, J, h, beta)

        burn_m = iterations // 2
        grids_m = chain_m[burn_m:]
        snapshot_grids.append(grids_m[-1])
        mags_m = np.array([magnetization(g) for g in grids_m])
        energies_m = np.array([energy_per_spin(g, J, h) for g in grids_m])

        avg_mags_m.append(np.mean(np.abs(mags_m)))
        avg_energies_m.append(np.mean(energies_m))

        # ---------- emcee ----------
        sampler = run_emcee(
            N_walkers=emcee_walkers,
            N_grid=N,
            J=J,
            h=h,
            beta=beta,
            seed=None,
            steps=iterations,
        )
        burn_e = iterations // 2
        chain_e = np.sign(sampler.get_chain()[burn_e:, :, :])

        mags_samples = chain_e.mean(axis=2)
        avg_mags_e.append(np.mean(np.abs(mags_samples)))

        final_states = chain_e[-1, :, :]
        energies_e_samples = []
        for s in final_states:
            grid = list_to_grid(s)
            energies_e_samples.append(energy_per_spin(grid, J, h))
        avg_energies_e.append(np.mean(energies_e_samples))


    mags_res_m = np.array(avg_mags_m)
    energies_res_m = np.array(avg_energies_m)
    mags_res_e = np.array(avg_mags_e)
    energies_res_e = np.array(avg_energies_e)

    plt.figure()
    plt.plot(betas, mags_res_m, label="Metropolis")
    plt.plot(betas, mags_res_e, label="emcee")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle |m| \rangle$")
    plt.title("Average Magnetization vs Beta: Metropolis vs emcee")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mag_vs_beta_comp.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(betas, energies_res_m, label="Metropolis")
    plt.plot(betas, energies_res_e, label="emcee")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle e \rangle$")
    plt.title("Average Energy per Spin vs Beta: Metropolis vs emcee")
    plt.legend()
    plt.tight_layout()
    plt.savefig("energy_vs_beta_comp.png", dpi=200, bbox_inches="tight")
    plt.close()

    plot_spin_snapshots(betas, snapshot_grids, N, filename="ising_snapshots.png")



if __name__ == "__main__":
    run_simulation(N=50, beta=1.0, J=1.0, h=0.0, iterations=100000, burn_frac=0.2, seed=123)
    #temperature_scan(N=20, beta_min=0.1, beta_max=1.0, J=1.0, h=0.0, iterations=100000, seed=456)


    
