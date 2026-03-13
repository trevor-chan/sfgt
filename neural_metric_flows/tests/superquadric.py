import math
import torch


def spow(x, e):
    """signed power used in superquadrics"""
    return torch.sign(x) * torch.abs(x) ** e


def superquadric(u, v, a=1., b=1., c=1., e1=1., e2=1.):
    cu, su = torch.cos(u), torch.sin(u)
    cv, sv = torch.cos(v), torch.sin(v)

    x = a * spow(cu, e1) * spow(cv, e2)
    y = b * spow(cu, e1) * spow(sv, e2)
    z = c * spow(su, e1)

    return torch.stack([x, y, z], dim=-1)


def compute_fundamental_forms(u, v, a=1., b=1., c=1., e1=1., e2=1.):
    """
    u,v : 1D tensors of length N
    returns:
        g : (N,2,2) metric tensor
        b : (N,2,2) second fundamental form
    """

    # This helper is sometimes used from endpoint-conditioned models during
    # evaluation. Re-enable grad locally so its analytic derivatives still work
    # even if the caller wrapped the outer forward pass in `torch.no_grad()`.
    with torch.enable_grad():
        u = u.clone().requires_grad_(True)
        v = v.clone().requires_grad_(True)

        X = superquadric(u, v, a, b, c, e1, e2)

        Xu = []
        Xv = []

        # compute first derivatives
        for i in range(3):
            grad_u = torch.autograd.grad(
                X[:, i].sum(), u, create_graph=True
            )[0]
            grad_v = torch.autograd.grad(
                X[:, i].sum(), v, create_graph=True
            )[0]

            Xu.append(grad_u)
            Xv.append(grad_v)

        Xu = torch.stack(Xu, dim=-1)
        Xv = torch.stack(Xv, dim=-1)

        # metric tensor
        E = (Xu * Xu).sum(dim=-1)
        F = (Xu * Xv).sum(dim=-1)
        G = (Xv * Xv).sum(dim=-1)

        g = torch.stack([
            torch.stack([E, F], dim=-1),
            torch.stack([F, G], dim=-1)
        ], dim=-2)

        # normal
        n = torch.cross(Xu, Xv, dim=-1)
        n = n / torch.norm(n, dim=-1, keepdim=True)

        # second derivatives
        Xuu = []
        Xuv = []
        Xvv = []

        for i in range(3):
            du = torch.autograd.grad(
                Xu[:, i].sum(), u, create_graph=True
            )[0]

            dv = torch.autograd.grad(
                Xu[:, i].sum(), v, create_graph=True
            )[0]

            dv2 = torch.autograd.grad(
                Xv[:, i].sum(), v, create_graph=True
            )[0]

            Xuu.append(du)
            Xuv.append(dv)
            Xvv.append(dv2)

        Xuu = torch.stack(Xuu, dim=-1)
        Xuv = torch.stack(Xuv, dim=-1)
        Xvv = torch.stack(Xvv, dim=-1)

        # second fundamental form
        L = (Xuu * n).sum(dim=-1)
        M = (Xuv * n).sum(dim=-1)
        N2 = (Xvv * n).sum(dim=-1)

        b = torch.stack([
            torch.stack([L, M], dim=-1),
            torch.stack([M, N2], dim=-1)
        ], dim=-2)

    return g, b


def compute_fundamental_forms_spherical(theta, phi, n=6.0, detach=True):
    """
    Compute fundamental forms for superquadric in (θ, φ) coordinates.

    Parameters
    ----------
    theta : Tensor (N,)
        Co-latitude θ ∈ [0, π], angle from z-axis.
    phi : Tensor (N,)
        Azimuth φ ∈ [0, 2π].
    n : float
        Superquadric exponent. n=2 is sphere, large n approaches cube.
    detach : bool
        If True, detach outputs from computation graph (for evaluation).

    Returns
    -------
    E, F, G, L, M, N : tuple of Tensor (N,)
        First and second fundamental form components in (θ, φ) coordinates.

    Notes
    -----
    The superquadric parameterization has singularities at "cube edges" where
    sin/cos of θ or φ = 0. Avoid θ ∈ {0, π/2, π} and φ ∈ {0, π/2, π, 3π/2, 2π}.
    Safe region: θ ∈ [0.2, 1.4], φ ∈ [0.2, 1.4] (one octant).
    """
    # Convert to (u, v) coordinates used by superquadric()
    # u = latitude (π/2 - θ), v = longitude (φ)
    u = math.pi / 2 - theta
    v = phi

    # Exponent for |x|^n + |y|^n + |z|^n = 1
    exponent = 2.0 / n

    g_uv, b_uv = compute_fundamental_forms(u, v, e1=exponent, e2=exponent)

    # Coordinate transformation: du/dθ = -1, dv/dφ = 1
    # Jacobian J = [[-1, 0], [0, 1]], so g_θφ = J^T g_uv J
    # Result: E_θφ = E_uv, F_θφ = -F_uv, G_θφ = G_uv (same for second form)
    E = g_uv[:, 0, 0]
    F = -g_uv[:, 0, 1]
    G = g_uv[:, 1, 1]
    L = b_uv[:, 0, 0]
    M = -b_uv[:, 0, 1]
    N_coef = b_uv[:, 1, 1]

    if detach:
        return E.detach(), F.detach(), G.detach(), L.detach(), M.detach(), N_coef.detach()
    return E, F, G, L, M, N_coef
