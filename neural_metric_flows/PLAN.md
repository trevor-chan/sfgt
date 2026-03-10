# Neural Fundamental Form Parameterization — Revised Plan

## Concept

A neural network directly parameterizes the first and second fundamental forms
of a surface as functions of parameter coordinates and (optionally) time:

**2D (flat surfaces):**
$$
\mathcal{N}_\theta(t, u, v) \to (\varepsilon, \varphi, \gamma)
$$
Validity constraint: Gauss equation $K = 0$.

**3D (general surfaces):**
$$
\mathcal{N}_\theta(t, u, v) \to (E, F, G, L, M, N)
$$
Validity constraints: Gauss equation + Codazzi-Mainardi equations.

The network learns a trajectory of valid surfaces $t \in [0, 1]$ by minimizing
compatibility PDE residuals plus regularization terms.

---

## 1. Mathematical Background

### 1.1 The Fundamental Theorem of Surfaces (Bonnet's Theorem)

A surface in $\mathbb{R}^3$ is determined (up to rigid motion) by its first
fundamental form $\mathrm{I} = (E, F, G)$ and second fundamental form
$\mathrm{II} = (L, M, N)$, provided they satisfy:

1. **Gauss equation** (integrability of the metric):

$$
K = \frac{LN - M^2}{EG - F^2}
$$

where $K$ is also expressible purely from $\mathrm{I}$ and its derivatives
via the Brioschi formula (Theorema Egregium):

$$
K = \frac{1}{(EG - F^2)^2}
\left[
\det \begin{pmatrix}
-\tfrac{1}{2}E_{vv} + F_{uv} - \tfrac{1}{2}G_{uu} & \tfrac{1}{2}E_u & F_u - \tfrac{1}{2}E_v \\
F_v - \tfrac{1}{2}G_u & E & F \\
\tfrac{1}{2}G_v & F & G
\end{pmatrix}
- \det \begin{pmatrix}
0 & \tfrac{1}{2}E_v & \tfrac{1}{2}G_u \\
\tfrac{1}{2}E_v & E & F \\
\tfrac{1}{2}G_u & F & G
\end{pmatrix}
\right]
$$

The Gauss equation requires that these two expressions for $K$ agree.
Equivalently, the residual:

$$
R_{\text{Gauss}} = (LN - M^2) - K_{\text{Brioschi}} \cdot (EG - F^2) = 0
$$

2. **Codazzi-Mainardi equations** (compatibility of $\mathrm{I}$ and $\mathrm{II}$):

$$
L_v - M_u = L\,\Gamma^1_{12} + M\,(\Gamma^2_{12} - \Gamma^1_{11}) - N\,\Gamma^2_{11}
$$

$$
M_v - N_u = L\,\Gamma^1_{22} + M\,(\Gamma^2_{22} - \Gamma^1_{12}) - N\,\Gamma^2_{12}
$$

where the Christoffel symbols are computed from $\mathrm{I}$:

$$
\Gamma^k_{ij} = \frac{1}{2} g^{kl}\left(
\frac{\partial g_{il}}{\partial x^j} +
\frac{\partial g_{jl}}{\partial x^i} -
\frac{\partial g_{ij}}{\partial x^l}
\right)
$$

### 1.2 Special Case: 2D Flat Surfaces

For surfaces confined to $\mathbb{R}^2$ (i.e. $\mathrm{II} = 0$), only the
first fundamental form matters, and the sole constraint is:

$$
K_{\text{Brioschi}}(E, F, G) = 0
$$

This is exactly our existing Gauss equation from the SFG solver.

---

## 2. Network Architecture

### 2.1 FundamentalFormNet

```python
class FundamentalFormNet(nn.Module):
    """
    Maps (t, u, v) -> fundamental forms.

    For mode='2d': outputs (E, F, G) with E, G > 0 and EG - F² > 0
    For mode='3d': outputs (E, F, G, L, M, N) with E, G > 0 and EG - F² > 0
    """
```

**Key architectural choices:**

- **Positivity enforcement**: E and G must be positive, and det(I) > 0.
  Use softplus or exp on raw outputs for E, G. Then constrain F via
  $F = \tanh(f_{\text{raw}}) \cdot \sqrt{EG}$ to guarantee $|F| < \sqrt{EG}$.

- **Endpoint conditioning**: To satisfy $a(0) = a_o$ and $a(1) = a_f$
  exactly, use hard boundary encoding:
  $$
  a(t, u, v) = (1-t)\,a_o(u,v) + t\,a_f(u,v) + t(1-t)\,\text{NN}_\theta(t, u, v)
  $$
  The NN output vanishes at $t = 0$ and $t = 1$ by construction.

- **Activation**: SIREN (sinusoidal) or SiLU. Needs at least C² for the
  compatibility equations (which involve second derivatives of the
  fundamental forms).

- **Input encoding**: Optional Fourier feature encoding of (u, v) to help
  the network learn high-frequency spatial variations.

### 2.2 Architecture Variants

**Shared trunk + separate heads:**
```
(t, u, v) → [shared MLP] → hidden
                            ├→ [head_I]  → (E, F, G)      # first FF
                            └→ [head_II] → (L, M, N)      # second FF (3D only)
```

**Fully coupled:**
```
(t, u, v) → [MLP] → (E, F, G, L, M, N)
```

The shared trunk variant may train more stably since I and II have
different scales and sensitivities.

---

## 3. Geometry Computation Layer

All derivatives computed via torch.autograd from the network output.

### 3.1 Christoffel Symbols (from first fundamental form)

```python
def christoffel_symbols(E, F, G, uv):
    """
    Compute Γ^k_ij from E, F, G and their first derivatives.

    Uses autograd to get E_u, E_v, F_u, F_v, G_u, G_v.
    Returns dict of Γ^1_11, Γ^1_12, Γ^1_22, Γ^2_11, Γ^2_12, Γ^2_22.
    """
```

$$
\Gamma^1_{11} = \frac{GE_u - 2FF_u + FE_v}{2(EG-F^2)}, \quad \text{etc.}
$$

### 3.2 Gaussian Curvature (Brioschi formula)

```python
def gaussian_curvature_brioschi(E, F, G, uv):
    """
    K from first fundamental form only (Theorema Egregium).
    Requires second derivatives E_uu, E_vv, F_uv, G_uu, G_vv.
    """
```

### 3.3 Gauss Equation Residual

```python
def gauss_residual(E, F, G, L, M, N, uv):
    """
    R_Gauss = (LN - M²) - K_Brioschi * (EG - F²)
    Should be zero for valid surfaces.
    """
```

### 3.4 Codazzi-Mainardi Residuals

```python
def codazzi_residuals(E, F, G, L, M, N, uv):
    """
    Two residuals from the Codazzi-Mainardi equations.
    Requires first derivatives of L, M, N and Christoffel symbols.
    Returns (R_codazzi_1, R_codazzi_2).
    """
```

### 3.5 Green Strain

```python
def green_strain(E, F, G, E0, F0, G0):
    """
    Strain tensor S = (1/2) a_0^{-1} (a - a_0).
    Returns trace(S), det(S), and eigenvalues.
    """
```

---

## 4. Loss Functions

### 4.1 Compatibility Losses (Surface Validity)

These are the hard constraints that must be satisfied for the fundamental
forms to correspond to a valid surface.

```python
def gauss_equation_loss(E, F, G, L, M, N, uv):
    """L2 residual of the Gauss equation."""
    R = gauss_residual(E, F, G, L, M, N, uv)
    return torch.mean(R**2)

def codazzi_loss(E, F, G, L, M, N, uv):
    """L2 residual of both Codazzi-Mainardi equations."""
    R1, R2 = codazzi_residuals(E, F, G, L, M, N, uv)
    return torch.mean(R1**2 + R2**2)

def flatness_loss(E, F, G, uv):
    """For 2D case: K_Brioschi = 0."""
    K = gaussian_curvature_brioschi(E, F, G, uv)
    return torch.mean(K**2)
```

### 4.2 Geometric Regularization

```python
def target_curvature_loss(K, K_target):
    """Match prescribed Gaussian curvature field."""
    return torch.mean((K - K_target)**2)

def mean_curvature_loss(H, H_target):
    """Match prescribed mean curvature (e.g. H=0 for minimal surfaces)."""
    return torch.mean((H - H_target)**2)

def metric_matching_loss(E, F, G, E_target, F_target, G_target):
    """Match a target metric (e.g. at endpoints)."""
    return torch.mean((E - E_target)**2 + 2*(F - F_target)**2 + (G - G_target)**2)
```

### 4.3 Mechanical / Physical Regularization

```python
def elastic_energy_loss(E, F, G, E0, F0, G0, alpha=1.0, beta=1.0):
    """
    St. Venant-Kirchhoff elastic energy.
    alpha: bulk modulus (penalizes area change)
    beta: shear modulus (penalizes shape distortion)
    """
    tr_S, det_S = green_strain_invariants(E, F, G, E0, F0, G0)
    return torch.mean(alpha * tr_S**2 + beta * (tr_S**2 - 2*det_S))

def strain_rate_loss(E, F, G, t):
    """
    Penalize rapid metric changes: ||da/dt||².
    Encourages smooth trajectories in time.
    Requires autograd w.r.t. t input.
    """
    ...
```

### 4.4 Boundary / Endpoint Losses

```python
def endpoint_loss(model, uv_pts, a_target, t_val):
    """
    Soft BC: match fundamental forms at t=0 or t=1.
    Only needed if hard encoding is not used.
    """
    ...

def spatial_boundary_loss(model, boundary_uv, a_boundary, t):
    """
    Dirichlet BCs on the spatial boundary of the parameter domain.
    """
    ...
```

---

## 5. Module Structure

```
neural_surface/
├── model.py              # FundamentalFormNet (2D and 3D modes)
├── geometry.py           # Christoffel symbols, curvatures, strain (autograd)
├── losses.py             # All losses from §4
├── training.py           # Training loop, collocation sampling, scheduling
└── viz.py                # Metric field visualization, reconstruction + plotting
```

---

## 6. Training Workflow

```
1. Initialize FundamentalFormNet with hard endpoint BCs
2. For each training step:
   a. Sample collocation points (t_i, u_i, v_i)
      - t uniform or stratified in [0, 1]
      - (u, v) uniform or quasi-random in parameter domain
      - separate boundary samples for spatial BCs
   b. Forward: predict (E, F, G, [L, M, N]) at collocation points
   c. Geometry: compute K, Christoffels, Codazzi residuals via autograd
   d. Losses: weighted sum
      - w_gauss * gauss_loss            (compatibility)
      - w_codazzi * codazzi_loss        (compatibility, 3D only)
      - w_elastic * elastic_loss        (regularization)
      - w_smooth * strain_rate_loss     (regularization)
      - w_bc * boundary_loss            (if soft BCs)
   e. Backward + optimizer step (Adam or L-BFGS)
3. Log and visualize periodically
```

### Loss Scheduling

Early training: upweight compatibility (Gauss, Codazzi) to first find
valid surfaces. Later: blend in regularization to select among the
family of valid surfaces.

Alternatively, use a penalty method: start with small weights on
compatibility and increase over training, forcing the solution onto
the constraint manifold gradually.

---

## 7. Test Cases

### Test 1: 2D Flat Trajectory (Square → Circle)

- Mode: 2D (first fundamental form only)
- Network: $\mathcal{N}(t, u, v) \to (\varepsilon, \varphi, \gamma)$
- Hard BCs: identity metric at $t=0$, elliptic map metric at $t=1$
- Loss: flatness ($K=0$) + optional elastic regularization
- Validation: compare with Gauss-only Newton solver results
- **This directly replaces our PDAE solver with a learned approach**

### Test 2: 3D Sphere Interpolation

- Mode: 3D (both fundamental forms)
- Interpolate between a flat plane and a sphere
- Loss: Gauss + Codazzi compatibility
- Validation: K should be 0 → 1/R² smoothly

### Test 3: Minimal Surface

- Mode: 3D, single surface (no time)
- Loss: Gauss + Codazzi + mean curvature $H = 0$
- BCs: prescribed boundary curve (e.g. catenoid boundary)

### Test 4: Elastic Growth Trajectory

- Mode: 3D with time
- Interpolate between two surfaces with prescribed elastic energy
- Loss: Gauss + Codazzi + elastic energy
- **Most general case — closest to morphoelasticity applications**

---

## 8. Key Advantages Over the PDE Solver

1. **No finite differences**: all derivatives via autograd, eliminating
   discretization error and the associated truncation-vs-convergence tension.

2. **No Newton convergence issues**: gradient descent on a loss landscape
   rather than root-finding. The optimizer naturally balances competing
   objectives via loss weights rather than requiring simultaneous satisfaction.

3. **Continuous in time**: the network represents the full trajectory as a
   continuous function, not a sequence of discrete snapshots. No time-stepping.

4. **Flexible regularization**: easy to add or remove objectives (elastic
   energy, curvature targets, smoothness) without restructuring the solver.

5. **Scalable**: the same architecture handles 2D and 3D, flat and curved,
   single surfaces and trajectories.

## 9. Key Risks

1. **Spectral bias**: MLPs are slow to learn high-frequency components.
   Mitigated by Fourier features or SIREN activations.

2. **Autograd cost**: second derivatives of the fundamental forms require
   third-order autograd through the network. May be expensive per step.

3. **Loss landscape**: compatibility PDEs are nonlinear constraints being
   imposed as soft penalties. The optimizer may find local minima that
   approximately but not exactly satisfy compatibility.

4. **Metric positivity**: must be enforced architecturally (softplus/exp
   outputs) to avoid degenerate metrics during training.
