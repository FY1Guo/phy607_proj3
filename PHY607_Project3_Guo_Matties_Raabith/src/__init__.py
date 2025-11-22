def generate_trace_plots():
    import argparse
    from .main import run_simulation

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beta", type=float, help="Inverse temperature parameter", const=0.3
    )
    parser.add_argument(
        "--hFactor", type=float, help="External magnetic field strength h", const=1.0
    )
    parser.add_argument("--coupling", type=float, help="Coupling strength J", const=1.0)
    parser.add_argument("--gridSize", type=int, help="Grid side length", const=20)
    parser.add_argument(
        "--steps", type=int, help="Number of proposed moves to run for", const=100000
    )
    parser.add_argument(
        "--burn",
        type=float,
        help="Fraction of the chain to classify as burn-in",
        const=0.3,
    )

    args = parser.parse_args()

    beta = args.beta
    h = args.hFactor
    J = args.coupling
    walkers = args.walkers
    iterations = args.steps
    burn = args.burn

    run_simulation(
        N=gridSize, beta=beta, J=J, h=h, iterations=iterations, burn_frac=burn
    )


def generate_phase_transition_plot():
    import argparse
    from .main import temperature_scan

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minBeta", type=float, help="Minimum beta value to sweep over", const=0.1
    )
    parser.add_argument(
        "--maxBeta", type=float, help="Maximum beta value to sweep over", const=1.0
    )
    parser.add_argument("--gridSize", type=int, help="Grid side length", const=20)
    parser.add_argument(
        "--steps", type=int, help="Number of proposed moves to run for", const=100000
    )
    parser.add_argument(
        "--hFactor", type=float, help="External magnetic field strength h", const=0
    )
    parser.add_argument("--coupling", type=float, help="Coupling strength J", const=1.0)

    args = parser.parse_args()

    min_beta = args.minBeta
    max_beta = args.maxBeta
    gridSize = args.gridSize
    steps = args.steps
    h = args.hFactor
    J = args.coupling

    temperature_scan(
        N=gridSize, beta_min=min_beta, beta_max=max_beta, J=J, h=h, iterations=steps
    )


def generate_convergence_plots():
    import argparse
    from .convergence import generate_convergence_test_plots

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beta", type=float, help="Inverse temperature parameter", const=0.3
    )
    parser.add_argument(
        "--hFactor", type=float, help="External magnetic field strength h", const=1.0
    )
    parser.add_argument("--coupling", type=float, help="Coupling strength J", const=1.0)
    parser.add_argument("--gridSize", type=int, help="Grid side length", const=20)
    parser.add_argument(
        "--steps", type=int, help="Number of proposed moves to run for", const=100000
    )
    parser.add_argument(
        "--burn",
        type=float,
        help="Fraction of the chain to classify as burn-in",
        const=0.3,
    )

    args = parser.parse_args()

    beta = args.beta
    h = args.hFactor
    J = args.coupling
    walkers = args.walkers
    iterations = args.steps
    burn = args.burn

    generate_convergence_test_plots(
        N=gridSize, beta=beta, J=J, h=h, iterations=iterations, burn_frac=burn
    )
