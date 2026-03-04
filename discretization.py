"""
discretization.py — Spatial discretization for the stress-free growth PDAE solver.

Defines:
  - Sparse finite difference operators (first and second derivatives, central)
  - Grid flattening/unflattening utilities
  - Boundary index masks
  - Ghost point handling for Dirichlet BCs
  - Numerical derivative computation via sparse operators
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Dict


# =============================================================================
# Index utilities
# =============================================================================

def ij_to_flat(i: int, j: int, K: int) -> int:
    """Convert 2D grid index (i, j) to flat index using row-major (C) ordering."""
    return i * K + j


def flat_to_ij(idx: int, K: int) -> Tuple[int, int]:
    """Convert flat index to 2D grid index (i, j)."""
    return divmod(idx, K)


def flatten_field(field: np.ndarray) -> np.ndarray:
    """Flatten a (K, K) field to (K^2,) in row-major order."""
    return field.ravel()


def unflatten_field(vec: np.ndarray, K: int) -> np.ndarray:
    """Reshape a (K^2,) vector back to (K, K)."""
    return vec.reshape(K, K)


# =============================================================================
# Boundary masks
# =============================================================================

def boundary_mask(K: int) -> np.ndarray:
    """
    Boolean mask of shape (K, K) that is True on the 1-cell boundary
    (edges of the grid) and False in the interior.
    """
    mask = np.zeros((K, K), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def interior_mask(K: int) -> np.ndarray:
    """Boolean mask that is True on interior points (complement of boundary)."""
    return ~boundary_mask(K)


def interior_indices(K: int) -> np.ndarray:
    """Flat indices of interior grid points."""
    mask = interior_mask(K)
    return np.where(mask.ravel())[0]


def boundary_indices(K: int) -> np.ndarray:
    """Flat indices of boundary grid points."""
    mask = boundary_mask(K)
    return np.where(mask.ravel())[0]


# =============================================================================
# Sparse difference operators
# =============================================================================

def _build_D1_u(K: int, du: float) -> sp.csr_matrix:
    """
    First derivative in u (row direction), second-order central difference.

    (D_u f)_{i,j} = (f_{i+1,j} - f_{i-1,j}) / (2 du)

    At boundaries (i=0 and i=K-1), the stencil is set to zero (no one-sided
    differences — boundary values are prescribed via BCs).

    Returns sparse matrix of shape (K^2, K^2).
    """
    N = K * K
    rows, cols, vals = [], [], []

    for i in range(1, K - 1):
        for j in range(K):
            row = ij_to_flat(i, j, K)
            col_plus = ij_to_flat(i + 1, j, K)
            col_minus = ij_to_flat(i - 1, j, K)

            rows.extend([row, row])
            cols.extend([col_plus, col_minus])
            vals.extend([1.0 / (2.0 * du), -1.0 / (2.0 * du)])

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _build_D1_v(K: int, dv: float) -> sp.csr_matrix:
    """
    First derivative in v (column direction), second-order central difference.

    (D_v f)_{i,j} = (f_{i,j+1} - f_{i,j-1}) / (2 dv)
    """
    N = K * K
    rows, cols, vals = [], [], []

    for i in range(K):
        for j in range(1, K - 1):
            row = ij_to_flat(i, j, K)
            col_plus = ij_to_flat(i, j + 1, K)
            col_minus = ij_to_flat(i, j - 1, K)

            rows.extend([row, row])
            cols.extend([col_plus, col_minus])
            vals.extend([1.0 / (2.0 * dv), -1.0 / (2.0 * dv)])

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _build_D2_uu(K: int, du: float) -> sp.csr_matrix:
    """
    Second derivative in u, second-order central difference.

    (D_uu f)_{i,j} = (f_{i+1,j} - 2f_{i,j} + f_{i-1,j}) / du^2
    """
    N = K * K
    rows, cols, vals = [], [], []
    du2 = du * du

    for i in range(1, K - 1):
        for j in range(K):
            row = ij_to_flat(i, j, K)
            col_center = row
            col_plus = ij_to_flat(i + 1, j, K)
            col_minus = ij_to_flat(i - 1, j, K)

            rows.extend([row, row, row])
            cols.extend([col_plus, col_center, col_minus])
            vals.extend([1.0 / du2, -2.0 / du2, 1.0 / du2])

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _build_D2_vv(K: int, dv: float) -> sp.csr_matrix:
    """
    Second derivative in v, second-order central difference.

    (D_vv f)_{i,j} = (f_{i,j+1} - 2f_{i,j} + f_{i,j-1}) / dv^2
    """
    N = K * K
    rows, cols, vals = [], [], []
    dv2 = dv * dv

    for i in range(K):
        for j in range(1, K - 1):
            row = ij_to_flat(i, j, K)
            col_center = row
            col_plus = ij_to_flat(i, j + 1, K)
            col_minus = ij_to_flat(i, j - 1, K)

            rows.extend([row, row, row])
            cols.extend([col_plus, col_center, col_minus])
            vals.extend([1.0 / dv2, -2.0 / dv2, 1.0 / dv2])

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _build_D2_uv(K: int, du: float, dv: float) -> sp.csr_matrix:
    """
    Mixed second derivative, second-order central difference.

    (D_uv f)_{i,j} = (f_{i+1,j+1} - f_{i+1,j-1} - f_{i-1,j+1} + f_{i-1,j-1}) / (4 du dv)
    """
    N = K * K
    rows, cols, vals = [], [], []
    coeff = 1.0 / (4.0 * du * dv)

    for i in range(1, K - 1):
        for j in range(1, K - 1):
            row = ij_to_flat(i, j, K)

            rows.extend([row, row, row, row])
            cols.extend([
                ij_to_flat(i + 1, j + 1, K),
                ij_to_flat(i + 1, j - 1, K),
                ij_to_flat(i - 1, j + 1, K),
                ij_to_flat(i - 1, j - 1, K),
            ])
            vals.extend([coeff, -coeff, -coeff, coeff])

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def build_difference_operators(K: int, du: float, dv: float = None) -> Dict[str, sp.csr_matrix]:
    """
    Build all five sparse difference operators for a K×K grid.

    Parameters
    ----------
    K : int
        Grid points per direction.
    du : float
        Grid spacing in the u direction.
    dv : float, optional
        Grid spacing in the v direction. Defaults to du (uniform grid).

    Returns
    -------
    ops : dict
        Keys: 'D_u', 'D_v', 'D_uu', 'D_vv', 'D_uv'
        Values: sparse CSR matrices of shape (K^2, K^2).
    """
    if dv is None:
        dv = du

    return {
        'D_u': _build_D1_u(K, du),
        'D_v': _build_D1_v(K, dv),
        'D_uu': _build_D2_uu(K, du),
        'D_vv': _build_D2_vv(K, dv),
        'D_uv': _build_D2_uv(K, du, dv),
    }


# =============================================================================
# Vectorized operator construction (for larger K)
# =============================================================================

def build_difference_operators_fast(K: int, du: float, dv: float = None) -> Dict[str, sp.csr_matrix]:
    """
    Build difference operators using Kronecker products — much faster for large K.

    Uses the identity:
      D_u on 2D grid = D_1d ⊗ I_K     (differentiate along rows)
      D_v on 2D grid = I_K ⊗ D_1d     (differentiate along columns)
      D_uv = D_u @ D_v                 (composed, valid at interior points)

    Boundary rows are zeroed out to match the loop-based construction.

    Parameters and returns are identical to build_difference_operators.
    """
    if dv is None:
        dv = du

    I_K = sp.eye(K, format='csr')

    # 1D first derivative (central, K×K), with boundary rows zeroed
    d1_u = sp.diags([1.0, -1.0], [1, -1], shape=(K, K), format='lil') / (2.0 * du)
    d1_u[0, :] = 0
    d1_u[-1, :] = 0

    d1_v = sp.diags([1.0, -1.0], [1, -1], shape=(K, K), format='lil') / (2.0 * dv)
    d1_v[0, :] = 0
    d1_v[-1, :] = 0

    # 1D second derivative (central, K×K), with boundary rows zeroed
    d2_u = sp.diags([1.0, -2.0, 1.0], [1, 0, -1], shape=(K, K), format='lil') / du**2
    d2_u[0, :] = 0
    d2_u[-1, :] = 0

    d2_v = sp.diags([1.0, -2.0, 1.0], [1, 0, -1], shape=(K, K), format='lil') / dv**2
    d2_v[0, :] = 0
    d2_v[-1, :] = 0

    # Convert to CSR for efficient arithmetic
    d1_u = d1_u.tocsr()
    d1_v = d1_v.tocsr()
    d2_u = d2_u.tocsr()
    d2_v = d2_v.tocsr()

    # 2D operators via Kronecker products
    D_u = sp.kron(d1_u, I_K, format='csr')
    D_v = sp.kron(I_K, d1_v, format='csr')
    D_uu = sp.kron(d2_u, I_K, format='csr')
    D_vv = sp.kron(I_K, d2_v, format='csr')
    D_uv = D_u @ D_v  # equivalent to the 4-point stencil at interior

    return {
        'D_u': D_u,
        'D_v': D_v,
        'D_uu': D_uu,
        'D_vv': D_vv,
        'D_uv': D_uv,
    }


# =============================================================================
# Apply operators to fields
# =============================================================================

def apply_operator(op: sp.csr_matrix, field: np.ndarray, K: int) -> np.ndarray:
    """
    Apply a sparse operator to a 2D field.

    Parameters
    ----------
    op : sparse matrix (K^2, K^2)
    field : ndarray of shape (K, K)
    K : int

    Returns
    -------
    result : ndarray of shape (K, K)
    """
    return unflatten_field(op @ flatten_field(field), K)


def compute_all_derivatives(
    ops: Dict[str, sp.csr_matrix],
    field: np.ndarray,
    K: int,
) -> Dict[str, np.ndarray]:
    """
    Compute all first and second derivatives of a 2D field using the
    prebuilt sparse operators.

    Returns
    -------
    derivs : dict with keys 'u', 'v', 'uu', 'vv', 'uv', each (K, K).
    """
    flat = flatten_field(field)
    return {
        'u': unflatten_field(ops['D_u'] @ flat, K),
        'v': unflatten_field(ops['D_v'] @ flat, K),
        'uu': unflatten_field(ops['D_uu'] @ flat, K),
        'vv': unflatten_field(ops['D_vv'] @ flat, K),
        'uv': unflatten_field(ops['D_uv'] @ flat, K),
    }


# =============================================================================
# Boundary condition application
# =============================================================================

def apply_dirichlet_bc(
    field: np.ndarray,
    bc_values: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Overwrite boundary values of a (K, K) field with prescribed Dirichlet values.

    Parameters
    ----------
    field : ndarray (K, K)
        Field to modify (modified in place and returned).
    bc_values : ndarray (K, K)
        Array containing the desired boundary values (only boundary entries used).
    K : int

    Returns
    -------
    field : ndarray (K, K), with boundary updated.
    """
    field[0, :] = bc_values[0, :]
    field[-1, :] = bc_values[-1, :]
    field[:, 0] = bc_values[:, 0]
    field[:, -1] = bc_values[:, -1]
    return field


def build_bc_projection(K: int) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Build sparse projection matrices that separate interior and boundary DOFs.

    Returns
    -------
    P_int : sparse (n_int, K^2) — extracts interior DOFs from a flat vector
    P_bnd : sparse (n_bnd, K^2) — extracts boundary DOFs from a flat vector
    """
    int_idx = interior_indices(K)
    bnd_idx = boundary_indices(K)
    N = K * K

    n_int = len(int_idx)
    n_bnd = len(bnd_idx)

    P_int = sp.csr_matrix(
        (np.ones(n_int), (np.arange(n_int), int_idx)),
        shape=(n_int, N),
    )
    P_bnd = sp.csr_matrix(
        (np.ones(n_bnd), (np.arange(n_bnd), bnd_idx)),
        shape=(n_bnd, N),
    )
    return P_int, P_bnd


def restrict_operator_to_interior(
    op: sp.csr_matrix,
    K: int,
    P_int: sp.csr_matrix = None,
) -> sp.csr_matrix:
    """
    Restrict a (K^2, K^2) operator to act only on interior points,
    producing a (n_int, n_int) matrix.

    This is useful for building the reduced system where boundary values
    are known and moved to the RHS.

    Parameters
    ----------
    op : sparse (K^2, K^2)
    K : int
    P_int : sparse (n_int, K^2), optional. Built if not provided.

    Returns
    -------
    op_int : sparse (n_int, n_int)
    """
    if P_int is None:
        P_int, _ = build_bc_projection(K)
    return P_int @ op @ P_int.T


def bc_rhs_contribution(
    op: sp.csr_matrix,
    bc_vec: np.ndarray,
    K: int,
    P_int: sp.csr_matrix = None,
) -> np.ndarray:
    """
    Compute the RHS contribution from known boundary values.

    When we split the system A x = b into interior and boundary parts:
        A_int @ x_int = b - A_bnd @ x_bnd

    This function returns  -(P_int @ op @ bc_full)  where bc_full has the
    boundary values at boundary indices and zeros at interior indices.

    Parameters
    ----------
    op : sparse (K^2, K^2) — full operator
    bc_vec : ndarray (K^2,) — vector with boundary values filled in, zeros at interior
    K : int
    P_int : sparse (n_int, K^2), optional

    Returns
    -------
    rhs_bc : ndarray (n_int,)
    """
    if P_int is None:
        P_int, _ = build_bc_projection(K)
    return -(P_int @ op @ bc_vec)


def make_bc_vector(bc_field: np.ndarray, K: int) -> np.ndarray:
    """
    Create a flat vector with boundary values at boundary indices and zeros elsewhere.
    Used for computing BC contributions to the RHS.
    """
    vec = np.zeros(K * K)
    bnd_idx = boundary_indices(K)
    vec[bnd_idx] = flatten_field(bc_field)[bnd_idx]
    return vec


# =============================================================================
# Convenience: full discretization setup
# =============================================================================

def setup_discretization(K: int, du: float, dv: float = None, fast: bool = True):
    """
    Build all discretization infrastructure for a K×K grid.

    Parameters
    ----------
    K : int
        Grid points per direction.
    du : float
        Grid spacing in u.
    dv : float, optional
        Grid spacing in v (default: du).
    fast : bool
        Use Kronecker-product construction (recommended for K > 50).

    Returns
    -------
    disc : dict with keys:
        'K'          : int
        'du', 'dv'   : float
        'ops'        : dict of sparse operators ('D_u', 'D_v', 'D_uu', 'D_vv', 'D_uv')
        'P_int'      : sparse interior projection
        'P_bnd'      : sparse boundary projection
        'int_idx'    : flat indices of interior points
        'bnd_idx'    : flat indices of boundary points
        'n_int'      : number of interior points
        'n_bnd'      : number of boundary points
        'int_mask'   : boolean (K, K)
        'bnd_mask'   : boolean (K, K)
    """
    if dv is None:
        dv = du

    builder = build_difference_operators_fast if fast else build_difference_operators
    ops = builder(K, du, dv)

    P_int, P_bnd = build_bc_projection(K)
    int_idx = interior_indices(K)
    bnd_idx = boundary_indices(K)

    return {
        'K': K,
        'du': du,
        'dv': dv,
        'ops': ops,
        'P_int': P_int,
        'P_bnd': P_bnd,
        'int_idx': int_idx,
        'bnd_idx': bnd_idx,
        'n_int': len(int_idx),
        'n_bnd': len(bnd_idx),
        'int_mask': interior_mask(K),
        'bnd_mask': boundary_mask(K),
    }
