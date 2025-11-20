import numpy as np
import matplotlib.pyplot as plt

from mcmc import run_chain, posterior, proposal_func
from Ising import magnetization, energy_per_spin


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


def run_simulation(N, beta=1.0, J=1.0, h=0.0, iterations=10000, burn_frac=0.5, seed=123):
    rng = np.random.default_rng(seed)
    init_grid = rng.choice([-1, 1], size=(N, N))

    plot_grid(init_grid, f"Initial configuration (N={N})", "grid_initial.png")

    chain_m, logp_m = run_chain(iterations, init_grid, posterior, proposal_func, J, h, beta)

    burn_m = int(iterations * burn_frac)
    grids_m = chain_m[burn_m:]
    mags_m = np.array([magnetization(g) for g in grids_m])
    energies_m = np.array([energy_per_spin(g, J, h) for g in grids_m])

    plot_grid(grids_m[-1], f"Final configuration (MCMC, N={N})", "grid_final_mcmc.png")

    plt.figure()
    plt.plot(mags_m)
    plt.xlabel("MC step (post burn-in)")
    plt.ylabel("Magnetization per spin")
    plt.title("Magnetization trace (MCMC)")
    plt.tight_layout()
    plt.savefig("magnetization_trace_mcmc.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(energies_m)
    plt.xlabel("MC step (post burn-in)")
    plt.ylabel("Energy per spin")
    plt.title("Energy trace (MCMC)")
    plt.tight_layout()
    plt.savefig("energy_trace_mcmc.png", dpi=200, bbox_inches="tight")
    plt.close()


def temperature_scan(N, beta_min, beta_max, J=1.0, h=0.0, iterations=10000, seed=456):
    avg_mags = []
    avg_energies = []

    rng = np.random.default_rng(seed)
    betas = np.linspace(beta_min, beta_max, 10)

    for beta in betas:
        init_grid = rng.choice([-1, 1], size=(N, N))

        chain_m, logp_m = run_chain(iterations, init_grid, posterior, proposal_func, J, h, beta)

        burn_m = iterations // 2
        grids_m = chain_m[burn_m:]
        mags_m = np.array([magnetization(g) for g in grids_m])
        energies_m = np.array([energy_per_spin(g, J, h) for g in grids_m])

        avg_mags.append(np.mean(np.abs(mags_m)))
        avg_energies.append(np.mean(energies_m))

    mags_res = np.array(avg_mags)
    energies_res = np.array(avg_energies)

    plt.figure()
    plt.plot(betas, mags_res, marker="o")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle |m| \rangle$")
    plt.title("Average Magnetization vs Beta")
    plt.tight_layout()
    plt.savefig("mag_vs_beta.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(betas, energies_res, marker="o")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle e \rangle$")
    plt.title("Average Energy per Spin vs Beta")
    plt.tight_layout()
    plt.savefig("energy_vs_beta.png", dpi=200, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    #run_simulation(N=50, beta=1.0, J=1.0, h=0.0, iterations=300000, burn_frac=0.2, seed=123)
    temperature_scan(N=20, beta_min=0.1, beta_max=1.0, J=1.0, h=0.0, iterations=100000, seed=456)


    