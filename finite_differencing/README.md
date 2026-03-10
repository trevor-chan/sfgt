# SFG Solver — Stress-Free Growth Trajectories

Solves the 3-equation PDAE system (Gauss + two algebraic strain constraints)
for flat-surface morphing between a square and a circle, following the
formulation in the reference paper.

## Requirements

Python 3.10+, NumPy, SciPy, Matplotlib. No install step — just run from the
`sfg_solver/` directory.

```bash
pip install numpy scipy matplotlib pytest
```

## Quick start

```python
from solver import SolverConfig, solve_trajectory
from visualization import generate_all_plots

config = SolverConfig(
    K=25,           # grid points per axis (K×K grid)
    margin=0.15,    # inset from [0,1]² boundary (avoids corner singularity)
    dt=0.02,        # time step
    t_end=0.5,      # final time (1.0 = full square→circle)
    newton_tol=1e-4,
    verbose=True,
)

result = solve_trajectory(config)
paths = generate_all_plots(result, output_dir='./plots')
```

## Running tests

```bash
cd sfg_solver
python -m pytest tests/ -v
```

115 tests across 5 modules: geometry (27), discretization (35), assembly (18),
solver (21), visualization (13).

## Module overview

| File | What it does |
|---|---|
| `geometry.py` | Elliptic grid mapping, endpoint metrics, Gauss equation, strain eigenvalues, interpolation |
| `discretization.py` | Sparse finite difference operators (D_u, D_v, D_uu, D_vv, D_uv), boundary projection |
| `assembly.py` | Linearized Jacobian assembly (3K² × 3K²), Tikhonov regularization, Newton solve |
| `solver.py` | Time-marching loop, line search with det>0 enforcement, adaptive Δt |
| `visualization.py` | Surface reconstruction via Cholesky integration, mesh/metric/convergence plots |

## Key parameters

- **K**: Grid resolution. K=15–25 for testing, K=50+ for production. Memory scales as O(K²).
- **margin**: Grid inset. The elliptic map's Jacobian determinant vanishes at corners;
  `margin=0.15` keeps things well-conditioned, `margin=0.05` gives more coverage but
  stiffer numerics.
- **dt**: Time step. Smaller is more stable but slower. The solver uses the linearly
  interpolated metric at each t as the Newton initial guess.
- **newton_tol**: Convergence threshold on L∞ residual norm.
- **regularization**: Tikhonov parameter (default 1e-8). Handles the structural
  rank deficiency in the φ-block when the reference metric has f=0.

## Paper corrections

Two errors were found in the reference paper during implementation:

1. **Gauss equation (eq. 4/10)**: The last terms use ε_v²·φ where the Brioschi
   formula gives ε_v²·γ. The cross-term structure is also wrong. The corrected
   version shows clean second-order convergence; the original has O(1) residuals.

2. **Strain eigenvalue (eq. 7)**: Explicit matrix entries are incorrect. The
   implementation uses the trace/determinant approach instead, which is verified
   against finite differences.
