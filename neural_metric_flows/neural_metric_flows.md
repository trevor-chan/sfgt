# Neural Metric Flows

## A Framework for Learning Intrinsic Surface Trajectories via Differentiable Fundamental Forms

---

# Part I — Mathematical Foundations

## 1. Surfaces and Their Intrinsic Description

### 1.1 Parametric Surfaces in $\mathbb{R}^3$

A smooth surface $\mathcal{S}$ embedded in $\mathbb{R}^3$ can be described by a map from a parameter domain $\Omega \subset \mathbb{R}^2$ to three-dimensional space:

$$
\mathbf{r} : \Omega \to \mathbb{R}^3, \qquad \mathbf{r}(u, v) = \bigl(x(u,v),\; y(u,v),\; z(u,v)\bigr)
$$

The parameters $(u, v)$ serve as coordinates on the surface. At each point, the partial derivatives

$$
\mathbf{r}_u = \frac{\partial \mathbf{r}}{\partial u}, \qquad
\mathbf{r}_v = \frac{\partial \mathbf{r}}{\partial v}
$$

span the tangent plane $T_p\mathcal{S}$. The surface is regular wherever $\mathbf{r}_u \times \mathbf{r}_v \neq 0$, meaning the tangent vectors are linearly independent and the parameterization is non-degenerate.

### 1.2 The First Fundamental Form (Metric Tensor)

The **first fundamental form** encodes how distances and angles are measured on the surface. For an infinitesimal displacement $(du, dv)$ in parameter space, the corresponding arc length on the surface is:

$$
ds^2 = d\mathbf{r} \cdot d\mathbf{r} = E\,du^2 + 2F\,du\,dv + G\,dv^2
$$

where the coefficients are:

$$
E = \mathbf{r}_u \cdot \mathbf{r}_u, \qquad
F = \mathbf{r}_u \cdot \mathbf{r}_v, \qquad
G = \mathbf{r}_v \cdot \mathbf{r}_v
$$

In matrix form, the metric tensor is the $2 \times 2$ symmetric positive-definite matrix:

$$
a = \begin{pmatrix} E & F \\ F & G \end{pmatrix}
$$

The metric tensor is the central object of intrinsic geometry. It determines:

- **Arc length** along a curve $\gamma(t) = (u(t), v(t))$:
$$
L = \int \sqrt{E\dot{u}^2 + 2F\dot{u}\dot{v} + G\dot{v}^2}\; dt
$$

- **Area** of a region $D \subset \Omega$:
$$
A = \iint_D \sqrt{EG - F^2}\; du\,dv
$$

- **Angles** between tangent vectors via the inner product $\langle \mathbf{a}, \mathbf{b} \rangle = a^i a_{ij} b^j$.

The determinant $\det(a) = EG - F^2 > 0$ is required for the metric to be non-degenerate.

**Physical interpretation.** $E$ measures stretching along the $u$-direction, $G$ measures stretching along the $v$-direction, and $F$ measures the shearing between the two coordinate directions. When $E = G = 1$ and $F = 0$, the parameterization is isometric — parameter-space distances equal surface distances.

### 1.3 The Unit Normal and the Shape Operator

The unit normal to the surface is:

$$
\mathbf{n} = \frac{\mathbf{r}_u \times \mathbf{r}_v}{\|\mathbf{r}_u \times \mathbf{r}_v\|}
$$

The **shape operator** (or Weingarten map) $S_p : T_p\mathcal{S} \to T_p\mathcal{S}$ describes how the normal changes as we move along the surface:

$$
S_p(\mathbf{v}) = -D_\mathbf{v}\,\mathbf{n}
$$

where $D_\mathbf{v}$ is the directional derivative along the tangent vector $\mathbf{v}$. The shape operator is self-adjoint with respect to the first fundamental form, meaning $\langle S(\mathbf{u}), \mathbf{v} \rangle = \langle \mathbf{u}, S(\mathbf{v}) \rangle$ for all tangent vectors $\mathbf{u}, \mathbf{v}$. Its eigenvalues $\kappa_1, \kappa_2$ are the **principal curvatures** of the surface — the maximum and minimum normal curvatures at each point.

### 1.4 The Second Fundamental Form

The **second fundamental form** encodes the extrinsic curvature — how the surface bends in the ambient space $\mathbb{R}^3$. It is defined through the second derivatives of the embedding:

$$
\mathrm{II} = L\,du^2 + 2M\,du\,dv + N\,dv^2
$$

where:

$$
L = \mathbf{r}_{uu} \cdot \mathbf{n}, \qquad
M = \mathbf{r}_{uv} \cdot \mathbf{n}, \qquad
N = \mathbf{r}_{vv} \cdot \mathbf{n}
$$

In matrix form:

$$
b = \begin{pmatrix} L & M \\ M & N \end{pmatrix}
$$

The shape operator in coordinates is $S = a^{-1}b$, from which:

$$
\text{Gaussian curvature:}\quad K = \det(S) = \frac{LN - M^2}{EG - F^2}
$$

$$
\text{Mean curvature:}\quad H = \frac{1}{2}\operatorname{tr}(S) = \frac{EN - 2FM + GL}{2(EG - F^2)}
$$

**Physical interpretation.** The second fundamental form captures how the surface deviates from its tangent plane. $L$ describes curvature along $u$, $N$ along $v$, and $M$ describes twisting. A flat surface has $L = M = N = 0$ everywhere. A sphere of radius $R$ has $L = N = 1/R$, $M = 0$. A saddle surface has $LN - M^2 < 0$.

### 1.5 The Intrinsic-Extrinsic Distinction

Gauss's Theorema Egregium establishes that the Gaussian curvature $K$ depends only on the first fundamental form and its derivatives — not on how the surface is embedded. This means $K$ is an *intrinsic* quantity: a creature living on the surface could measure it without reference to the ambient space.

The mean curvature $H$, by contrast, is extrinsic: it depends on how the surface is bent in $\mathbb{R}^3$. Two surfaces can have the same metric (same $E, F, G$) but different bending ($L, M, N$), provided the Gaussian curvature remains the same.

This distinction is central to Neural Metric Flows: the first fundamental form encodes the intrinsic geometry (distances, areas, angles), while the second fundamental form adds the extrinsic geometry (curvature, bending). Together, they determine the surface uniquely up to rigid motion.

---

## 2. Compatibility Equations

### 2.1 The Gauss Equation

Not every pair of symmetric bilinear forms $(a, b)$ corresponds to a valid surface. The first compatibility condition is the **Gauss equation**, which requires that the Gaussian curvature computed from the second fundamental form agrees with the curvature computed intrinsically from the metric alone.

The intrinsic curvature is given by the Brioschi formula:

$$
K_{\text{intr}} = \frac{1}{(EG - F^2)^2}\left[\det\begin{pmatrix} -\frac{1}{2}E_{vv} + F_{uv} - \frac{1}{2}G_{uu} & \frac{1}{2}E_u & F_u - \frac{1}{2}E_v \\ F_v - \frac{1}{2}G_u & E & F \\ \frac{1}{2}G_v & F & G \end{pmatrix} - \det\begin{pmatrix} 0 & \frac{1}{2}E_v & \frac{1}{2}G_u \\ \frac{1}{2}E_v & E & F \\ \frac{1}{2}G_u & F & G \end{pmatrix}\right]
$$

The extrinsic curvature is:

$$
K_{\text{extr}} = \frac{LN - M^2}{EG - F^2}
$$

The Gauss equation demands:

$$
K_{\text{extr}} = K_{\text{intr}}
$$

or equivalently:

$$
R_{\text{Gauss}} = (LN - M^2) - K_{\text{intr}} \cdot (EG - F^2) = 0
$$

For surfaces in $\mathbb{R}^2$ (flat surfaces), the second fundamental form vanishes identically ($L = M = N = 0$), and the Gauss equation reduces to the single constraint:

$$
K_{\text{intr}}(E, F, G) = 0
$$

This is the equation we solved numerically in the SFG solver. Its nonlinearity — involving products of first derivatives and second derivatives of the metric — is what makes flat-metric interpolation a nontrivial problem.

### 2.2 The Codazzi-Mainardi Equations

The second compatibility conditions are the **Codazzi-Mainardi equations**, which constrain how the second fundamental form varies across the surface:

$$
L_v - M_u = L\,\Gamma^1_{12} + M\,(\Gamma^2_{12} - \Gamma^1_{11}) - N\,\Gamma^2_{11}
$$

$$
M_v - N_u = L\,\Gamma^1_{22} + M\,(\Gamma^2_{22} - \Gamma^1_{12}) - N\,\Gamma^2_{12}
$$

These equations state that the "covariant derivative" of the second fundamental form is symmetric. They ensure that the bending described by $(L, M, N)$ is geometrically consistent with the intrinsic geometry described by $(E, F, G)$.

### 2.3 The Christoffel Symbols

The Christoffel symbols $\Gamma^k_{ij}$ encode how the coordinate basis vectors change as one moves across the surface. They are determined entirely by the first fundamental form:

$$
\Gamma^k_{ij} = \frac{1}{2}g^{kl}\left(\frac{\partial g_{il}}{\partial x^j} + \frac{\partial g_{jl}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^l}\right)
$$

where $g^{kl}$ are the components of the inverse metric $a^{-1}$. Explicitly, with $(x^1, x^2) = (u, v)$:

$$
\Gamma^1_{11} = \frac{GE_u - 2FF_u + FE_v}{2(EG - F^2)}, \qquad
\Gamma^1_{12} = \frac{GE_v - FG_u}{2(EG - F^2)}
$$

$$
\Gamma^1_{22} = \frac{2GF_v - GG_u - FG_v}{2(EG - F^2)}
$$

$$
\Gamma^2_{11} = \frac{2EF_u - EE_v - FE_u}{2(EG - F^2)}, \qquad
\Gamma^2_{12} = \frac{EG_u - FE_v}{2(EG - F^2)}
$$

$$
\Gamma^2_{22} = \frac{EG_v - 2FF_v + FG_u}{2(EG - F^2)}
$$

**Physical interpretation.** The Christoffel symbols play the role of the "gravitational field" in general relativity — they describe how "straight lines" (geodesics) curve in the coordinate system. On a flat surface in Cartesian coordinates, all Christoffel symbols vanish. On a curved surface or in curvilinear coordinates, they encode the connection between neighboring tangent planes.

### 2.4 Bonnet's Theorem (The Fundamental Theorem of Surfaces)

**Theorem (Bonnet).** Let $(E, F, G)$ and $(L, M, N)$ be smooth functions on a simply connected domain $\Omega$ such that $EG - F^2 > 0$ and the Gauss and Codazzi-Mainardi equations are satisfied. Then there exists a surface $\mathbf{r}: \Omega \to \mathbb{R}^3$ whose first and second fundamental forms are $(E, F, G)$ and $(L, M, N)$ respectively. This surface is unique up to rigid motions (translations and rotations) of $\mathbb{R}^3$.

This theorem is the foundation of Neural Metric Flows: by parameterizing $(E, F, G, L, M, N)$ as functions of $(t, u, v)$ and enforcing the compatibility equations, we obtain a valid family of surfaces without ever constructing the embedding. The rigid-motion ambiguity is a feature — we work with intrinsic quantities and recover the embedding only when needed for visualization.

---

## 3. Computing the Embedding from Fundamental Forms

Given valid fundamental forms $(E, F, G, L, M, N)$ satisfying the Gauss and Codazzi-Mainardi equations, reconstructing the surface $\mathbf{r}(u, v)$ requires solving a system of ODEs.

### 3.1 The Frame Equations (Gauss-Weingarten Relations)

The embedding is determined by an orthonormal frame $\{\mathbf{r}_u, \mathbf{r}_v, \mathbf{n}\}$ evolving according to:

$$
\frac{\partial}{\partial u}\begin{pmatrix}\mathbf{r}_u \\ \mathbf{r}_v \\ \mathbf{n}\end{pmatrix} = \begin{pmatrix} \Gamma^1_{11} & \Gamma^2_{11} & L \\ \Gamma^1_{12} & \Gamma^2_{12} & M \\ -\ell^1_1 & -\ell^2_1 & 0 \end{pmatrix}\begin{pmatrix}\mathbf{r}_u \\ \mathbf{r}_v \\ \mathbf{n}\end{pmatrix}
$$

$$
\frac{\partial}{\partial v}\begin{pmatrix}\mathbf{r}_u \\ \mathbf{r}_v \\ \mathbf{n}\end{pmatrix} = \begin{pmatrix} \Gamma^1_{12} & \Gamma^2_{12} & M \\ \Gamma^1_{22} & \Gamma^2_{22} & N \\ -\ell^1_2 & -\ell^2_2 & 0 \end{pmatrix}\begin{pmatrix}\mathbf{r}_u \\ \mathbf{r}_v \\ \mathbf{n}\end{pmatrix}
$$

where $\ell^i_j$ are the components of the shape operator $S = a^{-1}b$.

### 3.2 Reconstruction Procedure

1. **Choose initial conditions**: at a base point $(u_0, v_0)$, specify $\mathbf{r}(u_0, v_0)$ (position), $\mathbf{r}_u(u_0, v_0)$, $\mathbf{r}_v(u_0, v_0)$ (tangent vectors consistent with $E, F, G$), and $\mathbf{n}(u_0, v_0)$.

2. **Integrate the frame equations** outward from $(u_0, v_0)$, first along $u$ then $v$ (or any path covering $\Omega$).

3. **Recover the position** by integrating $\mathbf{r}_u$ and $\mathbf{r}_v$:
$$
\mathbf{r}(u, v) = \mathbf{r}(u_0, v_0) + \int_{u_0}^{u} \mathbf{r}_u(s, v_0)\,ds + \int_{v_0}^{v} \mathbf{r}_v(u, s)\,ds
$$

The Gauss and Codazzi-Mainardi equations guarantee path-independence: integrating along different paths to the same point yields the same result. This is precisely why these compatibility conditions are necessary.

### 3.3 The 2D (Flat) Case

For flat surfaces ($L = M = N = 0$), reconstruction simplifies substantially. The Gauss-Weingarten equations reduce to:

$$
\frac{\partial \mathbf{r}_u}{\partial u} = \Gamma^1_{11}\,\mathbf{r}_u + \Gamma^2_{11}\,\mathbf{r}_v, \qquad
\frac{\partial \mathbf{r}_u}{\partial v} = \Gamma^1_{12}\,\mathbf{r}_u + \Gamma^2_{12}\,\mathbf{r}_v
$$

Since the normal is constant (the surface lies in a plane), reconstruction requires only integrating these 2D frame equations plus a final position integration. This is the method used in the visualization module of the SFG solver, employing Christoffel-symbol-based parallel transport followed by a Poisson solve.

---

# Part II — Neural Metric Flows

## 4. Problem Statement

### 4.1 The Trajectory Problem

Given two surfaces $\mathcal{S}_0$ and $\mathcal{S}_1$ described by their fundamental forms:

$$
\text{Initial:}\quad a_0 = (E_0, F_0, G_0), \quad b_0 = (L_0, M_0, N_0)
$$

$$
\text{Final:}\quad a_1 = (E_1, F_1, G_1), \quad b_1 = (L_1, M_1, N_1)
$$

we seek a continuous family of fundamental forms:

$$
a(t, u, v) = \bigl(E(t,u,v),\; F(t,u,v),\; G(t,u,v)\bigr)
$$

$$
b(t, u, v) = \bigl(L(t,u,v),\; M(t,u,v),\; N(t,u,v)\bigr)
$$

for $t \in [0, 1]$, satisfying:

1. **Endpoint conditions**: $a(0) = a_0$, $b(0) = b_0$, $a(1) = a_1$, $b(1) = b_1$.
2. **Surface validity**: the Gauss and Codazzi-Mainardi equations hold at every $(t, u, v)$.
3. **Regularity**: $\det(a) > 0$ everywhere (non-degenerate metric).
4. **Problem-specific objectives**: minimized via regularization terms.

### 4.2 Non-Uniqueness and the Role of Regularization

The compatibility equations alone (conditions 1–3) typically admit infinitely many solutions. This is the space of all valid surface trajectories connecting $\mathcal{S}_0$ to $\mathcal{S}_1$. The specific trajectory we want depends on the physical or geometric context:

- **Geodesic trajectory**: minimizes a kinetic energy $\int_0^1 \|\dot{a}\|^2\,dt$, giving the "shortest" path through the space of metrics.

- **Minimal-strain trajectory**: minimizes integrated elastic energy, giving the path of least mechanical deformation.

- **Curvature-controlled trajectory**: maintains prescribed Gaussian or mean curvature along the path.

- **Growth trajectory**: follows a prescribed growth law where the rate of metric change is dictated by biological or physical processes.

These objectives enter as regularization terms in the neural network's training loss, selecting a unique solution from the family of valid trajectories.

### 4.3 Special Case: 2D Flat Trajectories

When both endpoints are flat ($K = 0$, $b = 0$), the problem reduces to:

$$
\mathcal{N}_\theta(t, u, v) \to \bigl(\varepsilon(t,u,v),\; \varphi(t,u,v),\; \gamma(t,u,v)\bigr)
$$

with the sole compatibility constraint being the Brioschi formula:

$$
K_{\text{Brioschi}}(\varepsilon, \varphi, \gamma) = 0 \qquad \forall\; t, u, v
$$

This is the setting of the stress-free growth problem, and the first test case for the Neural Metric Flows framework.

---

## 5. The Neural Metric Flow Architecture

### 5.1 Network Design

A neural network $\mathcal{N}_\theta$ parameterizes the fundamental forms as continuous functions of $(t, u, v)$:

$$
\mathcal{N}_\theta : [0, 1] \times \Omega \to \text{Sym}^+_2 \times \text{Sym}_2
$$

where $\text{Sym}^+_2$ is the cone of symmetric positive-definite $2 \times 2$ matrices (first fundamental form) and $\text{Sym}_2$ is the space of symmetric $2 \times 2$ matrices (second fundamental form).

**Input encoding.** The raw inputs $(t, u, v)$ may be passed through a Fourier feature mapping $\gamma(x) = [\sin(2\pi B x),\; \cos(2\pi B x)]$ to mitigate spectral bias, allowing the network to learn high-frequency spatial variations in the metric.

**Positivity enforcement.** The metric tensor must be positive definite ($E > 0$, $G > 0$, $EG - F^2 > 0$). This is enforced architecturally:

$$
E = \text{softplus}(E_{\text{raw}}), \qquad G = \text{softplus}(G_{\text{raw}})
$$

$$
F = \tanh(F_{\text{raw}}) \cdot \sqrt{EG} \cdot (1 - \epsilon)
$$

where $\epsilon$ is a small margin ensuring strict positive-definiteness. The second fundamental form $(L, M, N)$ is unconstrained.

**Hard endpoint conditioning.** To exactly satisfy the endpoint conditions without relying on soft penalties, the network output is structured as:

$$
a(t) = (1 - t)\,a_0 + t\,a_1 + t(1 - t)\,\mathcal{N}_\theta(t, u, v)
$$

The $t(1-t)$ factor ensures the correction vanishes identically at $t = 0$ and $t = 1$. The network learns only the deviation from linear interpolation, which is the part that must be adjusted to satisfy the compatibility equations.

**Activation functions.** The compatibility equations involve second spatial derivatives of the fundamental forms. Since the network output feeds into autograd for these derivatives, the activation functions must be at least $C^2$ (twice continuously differentiable). Suitable choices include SiLU ($x \cdot \sigma(x)$), Softplus ($\log(1 + e^x)$), or sinusoidal activations (SIREN).

### 5.2 Derivative Computation via Automatic Differentiation

All geometric quantities — Christoffel symbols, curvatures, Codazzi residuals — are computed via automatic differentiation through the network. This is a key advantage over finite-difference discretization:

- **Exact derivatives**: no truncation error from FD stencils.
- **Arbitrary resolution**: evaluation at any $(t, u, v)$ without re-discretization.
- **Natural backpropagation**: gradients of the loss with respect to $\theta$ flow through the geometry computation seamlessly.

The derivative chain is:

$$
\theta \xrightarrow{\text{forward}} (E, F, G, L, M, N) \xrightarrow{\text{autograd}} \text{Christoffel, curvatures, residuals} \xrightarrow{\text{loss}} \mathcal{L} \xrightarrow{\text{backward}} \nabla_\theta \mathcal{L}
$$

The Gauss equation requires second spatial derivatives of $(E, F, G)$, which involves second-order autograd through the network — differentiating the network output twice with respect to inputs. The Codazzi-Mainardi equations require first spatial derivatives of $(L, M, N)$ and first derivatives of $(E, F, G)$ (for Christoffel symbols), so only first-order autograd through the network.

---

## 6. Training Objective

### 6.1 Loss Structure

The total loss is a weighted sum:

$$
\mathcal{L}(\theta) = \underbrace{w_G \mathcal{L}_{\text{Gauss}} + w_C \mathcal{L}_{\text{Codazzi}}}_{\text{validity}} + \underbrace{w_{\text{BC}}\mathcal{L}_{\text{BC}}}_{\text{boundary}} + \underbrace{\sum_k w_k \mathcal{L}_k}_{\text{regularization}}
$$

The validity losses enforce that the network output corresponds to actual surfaces. The regularization losses select a specific trajectory from the family of valid solutions.

### 6.2 Validity Losses

**Gauss equation (3D):**

$$
\mathcal{L}_{\text{Gauss}} = \frac{1}{N}\sum_{i=1}^N \left[\frac{L_i N_i - M_i^2}{E_i G_i - F_i^2} - K_{\text{Brioschi}}(E_i, F_i, G_i)\right]^2
$$

**Flatness (2D special case):**

$$
\mathcal{L}_{\text{flat}} = \frac{1}{N}\sum_{i=1}^N K_{\text{Brioschi}}(E_i, F_i, G_i)^2
$$

**Codazzi-Mainardi (3D):**

$$
\mathcal{L}_{\text{Codazzi}} = \frac{1}{N}\sum_{i=1}^N \left(R^1_{\text{Cod},i}\right)^2 + \left(R^2_{\text{Cod},i}\right)^2
$$

where $R^1_{\text{Cod}}$ and $R^2_{\text{Cod}}$ are the two Codazzi residuals evaluated at collocation point $i$.

### 6.3 Regularization Losses

**Strain rate (temporal smoothness):**

$$
\mathcal{L}_{\text{rate}} = \frac{1}{N}\sum_i \left\|\frac{\partial a}{\partial t}\right\|^2_i = \frac{1}{N}\sum_i \left(\dot{E}_i^2 + 2\dot{F}_i^2 + \dot{G}_i^2\right)
$$

This penalizes rapid changes in the metric, encouraging temporally smooth trajectories. It is the simplest kinetic-energy-type term and corresponds to the $L^2$ (DeWitt) metric on the space of metrics. In the absence of other constraints, minimizing this alone (subject to endpoint conditions) yields the geodesic under the DeWitt metric, which decouples pointwise — each spatial point traces its own optimal path independently.

**Elastic energy (strain minimization):**

Given a reference metric $a_0 = (E_0, F_0, G_0)$, the Green strain tensor is $S = \frac{1}{2}a_0^{-1}(a - a_0)$. The St. Venant-Kirchhoff elastic energy is:

$$
\mathcal{L}_{\text{elastic}} = \frac{1}{N}\sum_i \left[\alpha\,(\operatorname{tr} S_i)^2 + \beta\,\operatorname{tr}(S_i^2)\right]
$$

The parameter $\alpha$ (bulk modulus) penalizes area change, and $\beta$ (shear modulus) penalizes shape distortion. This loss encourages trajectories that minimize the total elastic deformation relative to the reference state.

**Target Gaussian curvature:**

$$
\mathcal{L}_{K} = \frac{1}{N}\sum_i \left(K_i - K^*_i\right)^2
$$

where $K^*(t, u, v)$ is a prescribed curvature field. Setting $K^* = 0$ recovers the flatness loss. Setting $K^*(t) = t/R^2$ describes a gradual transition from flat to spherical.

**Target mean curvature:**

$$
\mathcal{L}_{H} = \frac{1}{N}\sum_i \left(H_i - H^*_i\right)^2
$$

Setting $H^* = 0$ yields a minimal surface at each time step. This is relevant for soap-film problems and membrane mechanics.

**Area preservation:**

$$
\mathcal{L}_{\text{area}} = \frac{1}{N}\sum_i \left(\sqrt{\det a_i} - \sqrt{\det a_{0,i}}\right)^2
$$

This enforces incompressible deformation — the surface deforms without changing local area. Relevant for thin membranes and biological tissues where material is conserved.

**Conformality:**

$$
\mathcal{L}_{\text{conf}} = \frac{1}{N}\sum_i \left[(E_i - G_i)^2 + 4F_i^2\right]
$$

This encourages a conformal (angle-preserving) parameterization, which is useful for visualization and for problems where angular distortion is more important than area distortion.

### 6.4 Loss Scheduling

The compatibility and regularization losses may have competing gradients during training. A practical strategy is **staged training**:

1. **Phase 1 — Compatibility**: Train primarily on $\mathcal{L}_{\text{Gauss}} + \mathcal{L}_{\text{Codazzi}}$ with small regularization weights to first find approximately valid surfaces.

2. **Phase 2 — Refinement**: Gradually increase regularization weights to select the desired trajectory from the valid family while maintaining compatibility.

3. **Phase 3 — Fine-tuning**: Use a higher-order optimizer (L-BFGS) or reduced learning rate to polish the solution.

Alternatively, an **augmented Lagrangian** approach treats compatibility as constraints and regularization as the objective, systematically driving constraint violation to zero while optimizing the objective.

---

## 7. Collocation and Sampling

### 7.1 Physics-Informed Training

Following the PINN (Physics-Informed Neural Network) paradigm, the PDE residuals are evaluated at **collocation points** — randomly sampled points in the domain $[0, 1] \times \Omega$ — rather than on a fixed grid. This offers several advantages:

- **No mesh generation**: the method is meshfree.
- **Adaptive resolution**: points can be concentrated in regions of high residual.
- **Stochastic regularization**: random resampling at each step acts as a form of regularization, preventing overfitting to a particular discretization.

### 7.2 Sampling Strategies

**Uniform random**: simplest approach, $t \sim U[0,1]$, $(u,v) \sim U[\Omega]$.

**Stratified**: divide $[0,1] \times \Omega$ into strata and sample within each, reducing variance.

**Residual-adaptive**: periodically evaluate the compatibility residuals on a fine grid and concentrate future samples in regions of high residual.

**Boundary-enriched**: sample extra points near $\partial\Omega$ and near $t = 0, 1$ to better enforce boundary conditions.

---

# Part III — Interpretation and Applications

## 8. Interpreting Metric and Shape Trajectories

### 8.1 What the Metric Trajectory Tells Us

A trajectory $a(t)$ through the space of metrics encodes a continuous deformation of the surface's intrinsic geometry. At each point $(u, v)$, the $2 \times 2$ metric tensor has two eigenvalues $\lambda_1(t) \geq \lambda_2(t) > 0$, which describe the principal stretches of the surface at that point relative to the parameter domain.

- **Isotropic expansion/contraction**: $\lambda_1 \approx \lambda_2$, both changing together. The surface grows or shrinks uniformly.
- **Shearing**: $\lambda_1 / \lambda_2$ changing, meaning the surface is being stretched more in one direction than another.
- **The determinant** $\det(a) = \lambda_1 \lambda_2$ measures local area change.
- **The trace** $\operatorname{tr}(a) = \lambda_1 + \lambda_2$ measures total stretching.

### 8.2 What the Shape Trajectory Tells Us

A trajectory $b(t)$ through the space of second fundamental forms describes how bending evolves. The principal curvatures $\kappa_1, \kappa_2$ (eigenvalues of $a^{-1}b$) tell us:

- **$\kappa_1 \kappa_2 > 0$ (synclastic)**: the surface curves like a sphere (dome or bowl).
- **$\kappa_1 \kappa_2 < 0$ (anticlastic)**: the surface curves like a saddle.
- **$\kappa_1 \kappa_2 = 0$ (developable)**: the surface curves along one direction only (cylinder, cone).
- **$\kappa_1 = \kappa_2$ (umbilic)**: the surface looks the same in all directions locally (sphere).

### 8.3 Interpreting Regularization Choices

Different regularization terms select different trajectories, each with a physical or geometric interpretation:

**Strain rate minimization** ($\mathcal{L}_{\text{rate}}$): Selects the trajectory that changes the metric most slowly. Analogous to a car taking the smoothest road between two cities. Physically corresponds to a deformation that proceeds at a uniform rate, minimizing acceleration-like effects. The resulting trajectory is a geodesic under the DeWitt metric.

**Elastic energy minimization** ($\mathcal{L}_{\text{elastic}}$): Selects the trajectory that stays closest to a reference configuration. Analogous to a rubber sheet being deformed — it follows the path of least elastic strain. The reference metric determines what "undeformed" means. This is the natural choice for mechanical problems where material properties matter.

**Curvature targeting** ($\mathcal{L}_K$, $\mathcal{L}_H$): Constrains the shape of intermediate surfaces. Relevant when the deformation path must pass through surfaces of specific geometry — for instance, a flat sheet that must inflate into a sphere while passing through a cap of prescribed curvature.

**Area preservation** ($\mathcal{L}_{\text{area}}$): Enforces incompressibility of the surface. Natural for biological membranes, fluid films, and thin sheets where material is conserved during deformation.

The key insight is that the compatibility equations (Gauss + Codazzi) define a *manifold* of valid surface trajectories, and the regularization selects a particular curve on this manifold. The neural network explores this manifold via gradient descent, and the loss landscape guides it toward the desired solution.

---

## 9. Applications and Analogies

### 9.1 Biological Growth and Morphogenesis

Biological tissues grow by locally adding material, which changes the intrinsic metric of the tissue. The central framework of **morphoelasticity** decomposes the deformation gradient as $\mathbf{F} = \mathbf{E} \cdot \mathbf{G}$, where $\mathbf{G}$ is a growth tensor prescribing the target metric and $\mathbf{E}$ is the elastic response. The grown metric $\mathbf{G}^T\mathbf{G}$ may be incompatible — it may have nonzero curvature — meaning no stress-free embedding exists in $\mathbb{R}^3$.

**Example: leaf ruffling.** A flat leaf with excess growth at the margins develops a metric with negative Gaussian curvature at the edges. The leaf cannot remain flat and must buckle into 3D, producing the characteristic wavy edges of lettuce or kale. A Neural Metric Flow could model this process by starting from a flat metric and prescribing a growth law that increases area at the boundary faster than the center. The compatibility equations determine whether the grown surface remains flat or must curve, and the elastic regularization determines the equilibrium shape.

**Example: gut tube looping.** The embryonic gut starts as a straight tube and loops into a complex 3D shape through differential growth. The metric tensor of the gut wall changes non-uniformly, creating curvature and torsion. A time-varying Neural Metric Flow could parameterize the evolution of the gut's geometry from tube to loops, with the growth law providing the driving force and elastic energy selecting the physical equilibrium.

### 9.2 Shape Morphing and Computer Graphics

Interpolating between two 3D shapes is a fundamental problem in computer graphics and geometric modeling. Traditional approaches work with vertex positions or displacement fields, but the metric formulation offers advantages:

- **Resolution independence**: the metric is a continuous field, not tied to a mesh resolution.
- **Intrinsic control**: you can prescribe how stretching and bending change along the morph, independent of the ambient-space path.
- **Natural distortion measures**: elastic energy and curvature provide physically meaningful quality metrics for the morph.

A Neural Metric Flow between a cube and a sphere, for instance, would find fundamental forms that smoothly transition between the two geometries while satisfying the compatibility equations and minimizing some notion of distortion.

### 9.3 Computational Anatomy

In medical image analysis, comparing anatomical shapes (brain surfaces, organ boundaries) across individuals requires establishing correspondences. The LDDMM framework computes diffeomorphisms via velocity fields in the embedding space. A metric-based approach would instead characterize the deformation by how the intrinsic geometry of the anatomical surface changes — how tissue stretches, shears, and curves.

The advantage is that metric-space descriptors are intrinsic: they don't depend on the patient's orientation in the scanner or on arbitrary choices of ambient coordinate system. The strain tensor between two brain surfaces directly measures local tissue deformation, which is what neuroscientists care about when studying atrophy, growth, or injury.

### 9.4 Soft Robotics and Programmable Materials

Designing materials that morph into prescribed shapes on command (via heat, light, swelling, or electric fields) requires solving an inverse problem: given a target shape, what pattern of local expansion/contraction achieves it? This is the inverse of the morphoelastic forward problem.

A trained Neural Metric Flow could serve as a fast surrogate model: given an initial flat sheet and a target 3D shape, predict the growth tensor field (metric trajectory) that achieves the target. The compatibility equations ensure physical realizability, and the elastic regularization ensures the trajectory is mechanically feasible.

### 9.5 General Relativity and Cosmology

In general relativity, spacetime is a 4D Lorentzian manifold whose metric evolves according to the Einstein field equations. The ADM formalism decomposes this into a spatial metric $g_{ij}(t, \mathbf{x})$ evolving in time — exactly the structure of a Neural Metric Flow, but with the Gauss and Codazzi equations replaced by the Hamiltonian and momentum constraints of GR. While the physics is different, the computational framework is analogous: parameterize the spatial metric as a neural field, enforce the constraint equations as losses, and use regularization to select physically meaningful solutions.

---

# Part IV — Computational Framework

## 10. Code Architecture

```
neural_metric_flows/
│
├── geometry.py           # Differential geometry via autograd
│   ├── christoffel_symbols(E, F, G, uv)
│   ├── gaussian_curvature_brioschi(E, F, G, uv)
│   ├── gauss_residual_3d(E, F, G, L, M, N, uv)
│   ├── codazzi_residuals(E, F, G, L, M, N, uv)
│   ├── gaussian_curvature(E, F, G, L, M, N)
│   ├── mean_curvature(E, F, G, L, M, N)
│   └── green_strain_invariants(E, F, G, E0, F0, G0)
│
├── model.py              # Network architectures
│   ├── FundamentalFormNet     (t, u, v) → (E, F, G, [L, M, N])
│   ├── enforce_metric_positivity
│   └── SirenLayer
│
├── losses.py             # Loss function library
│   ├── Validity: flatness_loss, gauss_equation_loss, codazzi_loss
│   ├── Geometric: target_curvature_loss, conformality_loss
│   ├── Mechanical: elastic_energy_loss, strain_rate_loss
│   └── Boundary: endpoint_metric_loss, spatial_boundary_loss
│
├── training.py           # Training loop and utilities
│   ├── sample_collocation(n, domain)
│   ├── train_step(model, optimizer, loss_config)
│   └── train(model, config, n_steps)
│
├── reconstruction.py     # Embedding recovery from fundamental forms
│   ├── reconstruct_frame(E, F, G, L, M, N)
│   └── integrate_position(frame)
│
└── viz.py                # Visualization
    ├── plot_metric_fields(results)
    ├── plot_curvature_fields(results)
    └── plot_3d_surface(embedding)
```

## 11. Workflow Summary

1. **Define the problem**: specify endpoint surfaces (their fundamental forms), choose which constraints and regularizers apply.

2. **Build the network**: instantiate `FundamentalFormNet` with appropriate mode (2D/3D), endpoint conditioning, and architecture choices.

3. **Configure the loss**: select weights for compatibility terms (Gauss, Codazzi) and regularization terms (elastic, strain rate, curvature).

4. **Train**: sample collocation points, compute losses via autograd geometry, optimize via Adam or L-BFGS.

5. **Evaluate**: measure compatibility residuals on a dense grid, compute geometric quantities (curvature, strain, area change).

6. **Reconstruct** (optional): solve the frame equations to obtain the 3D embedding for visualization.

---

## 12. Summary of Key Ideas

Neural Metric Flows represent surface deformations through time-varying fundamental forms parameterized by neural networks. The framework rests on three pillars:

**Bonnet's theorem** guarantees that satisfying the Gauss and Codazzi-Mainardi equations is necessary and sufficient for the fundamental forms to correspond to a valid surface. The compatibility equations serve as the "physics" of the framework.

**Automatic differentiation** replaces finite differences for computing all geometric quantities, eliminating discretization error and enabling meshfree, resolution-independent computation. The compatibility PDEs are evaluated as pointwise losses at collocation points.

**Soft-constraint optimization** replaces Newton-based root-finding. Instead of demanding exact simultaneous satisfaction of nonlinear PDE constraints, the neural network is trained via gradient descent on a loss landscape that balances validity against problem-specific objectives. The implicit regularization of neural networks — smooth, low-rank representations — acts as a natural prior that biases the solution toward physically reasonable surfaces.

The result is a framework that can parameterize continuous families of surfaces, enforce geometric validity, and optimize for application-specific objectives — from elastic growth trajectories to shape morphing to minimal surfaces — all within a unified computational architecture.
