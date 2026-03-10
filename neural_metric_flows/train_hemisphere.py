"""
train_hemisphere.py — Learn a 3D surface trajectory from flat sheet to hemisphere.

Network: (t, u, v) -> (E, F, G, L, M, N)
Hard BCs: flat sheet at t=0, hemisphere at t=1
Loss: Gauss + Codazzi compatibility + curvature regularization
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

from model import FundamentalFormNet
from losses import compatibility_loss_3d, elastic_energy_loss, strain_rate_loss


# =============================================================================
# Endpoint metrics
# =============================================================================

def metric_flat_sheet(uv):
    """
    Flat sheet: identity metric, zero curvature.
    
    First fundamental form: E=1, F=0, G=1
    Second fundamental form: L=0, M=0, N=0
    """
    N = uv.shape[0]
    device, dtype = uv.device, uv.dtype
    
    E = torch.ones(N, device=device, dtype=dtype)
    F = torch.zeros(N, device=device, dtype=dtype)
    G = torch.ones(N, device=device, dtype=dtype)
    L = torch.zeros(N, device=device, dtype=dtype)
    M = torch.zeros(N, device=device, dtype=dtype)
    N_coef = torch.zeros(N, device=device, dtype=dtype)
    
    return (E, F, G, L, M, N_coef)


def metric_hemisphere(uv):
    """
    Unit hemisphere as Monge patch: z = sqrt(1 - u² - v²).
    
    First fundamental form:
        E = (1 - v²) / z²
        F = uv / z²
        G = (1 - u²) / z²
        
    Second fundamental form (shape operator = identity for unit sphere):
        L = E, M = F, N = G
    """
    u, v = uv[:, 0], uv[:, 1]
    u2, v2 = u * u, v * v
    
    # z² = 1 - u² - v², need to stay inside the unit disk
    z2 = 1.0 - u2 - v2
    z2 = torch.clamp(z2, min=1e-6)  # safety for points near boundary
    
    # First fundamental form
    E = (1.0 - v2) / z2
    F = (u * v) / z2
    G = (1.0 - u2) / z2
    
    # Second fundamental form (for unit hemisphere, shape operator is identity)
    # This gives K = 1, H = 1 everywhere
    L = E.clone()
    M = F.clone()
    N_coef = G.clone()
    
    return (E, F, G, L, M, N_coef)


# =============================================================================
# Sampling
# =============================================================================

def sample_collocation(n_points, lo=-0.7, hi=0.7, device='cpu'):
    """
    Sample (t, u, v) uniformly.
    
    Use a smaller domain than [-1,1] to stay well inside the unit disk
    for the hemisphere parameterization.
    """
    t = torch.rand(n_points, 1, device=device)
    uv = lo + (hi - lo) * torch.rand(n_points, 2, device=device)
    tuv = torch.cat([t, uv], dim=1)
    tuv.requires_grad_(True)
    return tuv


# =============================================================================
# Training
# =============================================================================

def train(
    n_steps=10000,
    n_collocation=2048,
    hidden_dim=128,
    n_layers=5,
    activation='silu',
    lr=1e-3,
    w_compat=1.0,
    w_elastic=0.0,
    w_strain_rate=0.01,
    margin=0.3,
    log_every=500,
    device='cpu',
):
    """
    Train neural fundamental forms for sheet-to-hemisphere transformation.
    """
    lo = -1.0 + margin
    hi = 1.0 - margin

    model = FundamentalFormNet(
        mode='3d',
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation=activation,
        endpoint_a_o=metric_flat_sheet,
        endpoint_a_f=metric_hemisphere,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    history = {
        'loss_total': [], 'loss_compat': [], 'loss_elastic': [],
        'loss_strain_rate': [], 'K_mean': [], 'H_mean': [],
    }

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params} parameters, {activation} activation")
    print(f"Training: {n_steps} steps, {n_collocation} collocation points")
    print(f"Domain: [{lo:.2f}, {hi:.2f}]² (margin={margin})")
    print(f"Weights: compat={w_compat}, elastic={w_elastic}, strain_rate={w_strain_rate}")
    print()

    t_start = time.time()

    for step in range(n_steps):
        optimizer.zero_grad()

        tuv = sample_collocation(n_collocation, lo=lo, hi=hi, device=device)

        E, F, G, L, M, N_coef = model(tuv)

        # Gauss + Codazzi compatibility loss
        L_compat = compatibility_loss_3d(E, F, G, L, M, N_coef, tuv, u_idx=1, v_idx=2)

        # Optional elastic regularization
        L_elastic = torch.tensor(0.0, device=device)
        if w_elastic > 0:
            uv = tuv[:, 1:3]
            E0, F0, G0, _, _, _ = metric_flat_sheet(uv)
            L_elastic = elastic_energy_loss(E, F, G, E0, F0, G0)

        # Optional strain rate regularization
        L_strain_rate = torch.tensor(0.0, device=device)
        if w_strain_rate > 0:
            L_strain_rate = strain_rate_loss(E, F, G, tuv)

        loss = w_compat * L_compat + w_elastic * L_elastic + w_strain_rate * L_strain_rate

        # Compute curvatures for monitoring
        det_a = E * G - F * F
        K = (L * N_coef - M * M) / (det_a + 1e-12)
        H = (E * N_coef + G * L - 2 * F * M) / (2 * det_a + 1e-12)
        K_mean = torch.mean(torch.abs(K)).detach().item()
        H_mean = torch.mean(torch.abs(H)).detach().item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        history['loss_total'].append(loss.item())
        history['loss_compat'].append(L_compat.item())
        history['loss_elastic'].append(L_elastic.item())
        history['loss_strain_rate'].append(L_strain_rate.item())
        history['K_mean'].append(K_mean)
        history['H_mean'].append(H_mean)

        if (step + 1) % log_every == 0 or step == 0:
            elapsed = time.time() - t_start
            print(f"Step {step+1:5d}/{n_steps}  "
                  f"loss={loss.item():.4e}  "
                  f"|K|={K_mean:.4f}  |H|={H_mean:.4f}  "
                  f"({elapsed:.1f}s)")

    print(f"\nTraining complete in {time.time() - t_start:.1f}s")
    return model, history


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, n_t=11, K_grid=20, margin=0.3, device='cpu'):
    """Evaluate trained model on a regular grid."""
    lo = -1.0 + margin
    hi = 1.0 - margin
    uv_1d = torch.linspace(lo, hi, K_grid, device=device)
    U, V = torch.meshgrid(uv_1d, uv_1d, indexing='ij')
    uv_flat = torch.stack([U.reshape(-1), V.reshape(-1)], dim=1)

    t_vals = torch.linspace(0, 1, n_t, device=device)
    results = []

    for t_val in t_vals:
        N = uv_flat.shape[0]
        t_col = torch.full((N, 1), t_val.item(), device=device)
        tuv = torch.cat([t_col, uv_flat], dim=1)

        with torch.no_grad():
            E, F, G, L, M, N_coef = model(tuv)

        det_a = E * G - F * F
        K = (L * N_coef - M * M) / (det_a + 1e-12)
        H = (E * N_coef + G * L - 2 * F * M) / (2 * det_a + 1e-12)

        results.append({
            't': t_val.item(),
            'E': E.reshape(K_grid, K_grid).cpu().numpy(),
            'F': F.reshape(K_grid, K_grid).cpu().numpy(),
            'G': G.reshape(K_grid, K_grid).cpu().numpy(),
            'L': L.reshape(K_grid, K_grid).cpu().numpy(),
            'M': M.reshape(K_grid, K_grid).cpu().numpy(),
            'N': N_coef.reshape(K_grid, K_grid).cpu().numpy(),
            'K': K.reshape(K_grid, K_grid).cpu().numpy(),
            'H': H.reshape(K_grid, K_grid).cpu().numpy(),
            'det': det_a.reshape(K_grid, K_grid).cpu().numpy(),
        })

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_results(results, history, save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].semilogy(history['loss_total'], label='Total', color='k')
    axes[0].semilogy(history['loss_compat'], label='Compatibility', color='C0', alpha=0.7)
    if max(history['loss_elastic']) > 0:
        axes[0].semilogy(history['loss_elastic'], label='Elastic', color='C1', alpha=0.7)
    if max(history['loss_strain_rate']) > 0:
        axes[0].semilogy(history['loss_strain_rate'], label='Strain rate', color='C2', alpha=0.7)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()

    axes[1].plot(history['K_mean'], label='|K| mean', color='C0')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Gaussian Curvature')
    axes[1].set_title('Mean |K| During Training')
    axes[1].legend()

    axes[2].plot(history['H_mean'], label='|H| mean', color='C1')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Mean Curvature')
    axes[2].set_title('Mean |H| During Training')
    axes[2].legend()

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()

    # Curvature evolution
    n_t = len(results)
    indices = sorted(set([0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]))

    fig, axes = plt.subplots(2, len(indices), figsize=(4 * len(indices), 7))
    for col, idx in enumerate(indices):
        r = results[idx]
        
        # Gaussian curvature K
        K = r['K']
        vmax_K = max(abs(K.min()), abs(K.max()), 0.1)
        im = axes[0, col].imshow(K, origin='lower', cmap='RdBu_r',
                                  vmin=-vmax_K, vmax=vmax_K)
        axes[0, col].set_title(f"K, t={r['t']:.2f}\n[{K.min():.2f}, {K.max():.2f}]")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)
        
        # Mean curvature H
        H = r['H']
        vmax_H = max(abs(H.min()), abs(H.max()), 0.1)
        im = axes[1, col].imshow(H, origin='lower', cmap='RdBu_r',
                                  vmin=-vmax_H, vmax=vmax_H)
        axes[1, col].set_title(f"H, t={r['t']:.2f}\n[{H.min():.2f}, {H.max():.2f}]")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046)

    axes[0, 0].set_ylabel('Gaussian K')
    axes[1, 0].set_ylabel('Mean H')
    plt.suptitle('Curvature Evolution (Sheet → Hemisphere)', fontsize=13)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'curvature_evolution.png'), dpi=150)
    plt.close()

    # First fundamental form evolution
    fig, axes = plt.subplots(3, len(indices), figsize=(4 * len(indices), 10))
    for col, idx in enumerate(indices):
        r = results[idx]
        for row, (field, name) in enumerate([
            (r['E'], 'E'), (r['F'], 'F'), (r['G'], 'G')
        ]):
            if name == 'F':
                vabs = max(abs(field.min()), abs(field.max()), 0.01)
                im = axes[row, col].imshow(field, origin='lower', cmap='RdBu_r',
                                            vmin=-vabs, vmax=vabs)
            else:
                im = axes[row, col].imshow(field, origin='lower', cmap='viridis')
            axes[row, col].set_title(f"{name}, t={r['t']:.2f}")
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)

    plt.suptitle('First Fundamental Form Evolution')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'first_fundamental_form.png'), dpi=150)
    plt.close()

    # Second fundamental form evolution
    fig, axes = plt.subplots(3, len(indices), figsize=(4 * len(indices), 10))
    for col, idx in enumerate(indices):
        r = results[idx]
        for row, (field, name) in enumerate([
            (r['L'], 'L'), (r['M'], 'M'), (r['N'], 'N')
        ]):
            vabs = max(abs(field.min()), abs(field.max()), 0.01)
            im = axes[row, col].imshow(field, origin='lower', cmap='RdBu_r',
                                        vmin=-vabs, vmax=vabs)
            axes[row, col].set_title(f"{name}, t={r['t']:.2f}")
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)

    plt.suptitle('Second Fundamental Form Evolution')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'second_fundamental_form.png'), dpi=150)
    plt.close()

    # Curvature vs time summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    t_vals = [r['t'] for r in results]
    K_max = [np.max(np.abs(r['K'])) for r in results]
    K_mean = [np.mean(r['K']) for r in results]
    H_max = [np.max(np.abs(r['H'])) for r in results]
    H_mean = [np.mean(r['H']) for r in results]
    
    axes[0].plot(t_vals, K_mean, 'o-', label='K mean', color='C0')
    axes[0].fill_between(t_vals, 
                         [np.min(r['K']) for r in results],
                         [np.max(r['K']) for r in results],
                         alpha=0.2, color='C0')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(1, color='green', linestyle='--', alpha=0.5, label='Target K=1')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('Gaussian Curvature K')
    axes[0].set_title('K evolution: 0 (flat) → 1 (hemisphere)')
    axes[0].legend()
    
    axes[1].plot(t_vals, H_mean, 'o-', label='H mean', color='C1')
    axes[1].fill_between(t_vals,
                         [np.min(r['H']) for r in results],
                         [np.max(r['H']) for r in results],
                         alpha=0.2, color='C1')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(1, color='green', linestyle='--', alpha=0.5, label='Target H=1')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('Mean Curvature H')
    axes[1].set_title('H evolution: 0 (flat) → 1 (hemisphere)')
    axes[1].legend()

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'curvature_summary.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {save_dir}/")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    save_dir = 'results_hemisphere'

    model, history = train(
        n_steps=5000,
        n_collocation=2048,
        hidden_dim=128,
        n_layers=5,
        activation='silu',
        lr=1e-3,
        w_compat=1.0,
        w_elastic=0.0,
        w_strain_rate=0.01,
        margin=0.5,  # Stay well inside unit disk for hemisphere
        log_every=500,
        device=device,
    )

    print("\nEvaluating on grid...")
    results = evaluate(model, n_t=11, K_grid=20, margin=0.5, device=device)

    for r in results:
        print(f"  t={r['t']:.2f}: K=[{r['K'].min():.3f}, {r['K'].max():.3f}]  "
              f"H=[{r['H'].min():.3f}, {r['H'].max():.3f}]")

    print("\nGenerating plots...")
    plot_results(results, history, save_dir=save_dir)

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    print(f"Model saved to {save_dir}/model.pt")
