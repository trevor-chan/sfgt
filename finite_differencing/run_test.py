from solver import SolverConfig, solve_trajectory
from visualization import generate_all_plots

config = SolverConfig(
    K=25,           # grid points per axis (K×K grid)
    margin=0.05,    # inset from [0,1]² boundary (avoids corner singularity)
    dt=0.005,        # time step
    t_end=1.0,      # final time (1.0 = full square→circle)
    newton_tol=1e-4,
    verbose=True,
)

result = solve_trajectory(config)
paths = generate_all_plots(result, output_dir='./plots')