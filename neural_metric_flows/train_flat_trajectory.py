"""
train_flat_trajectory.py — Test 1: Learn a flat metric trajectory from
square to circle using neural fundamental form parameterization.

Network: (t, u, v) -> (E, F, G)
Hard BCs: identity at t=0, elliptic map metric at t=1
Loss: flatness (K=0) + optional elastic/strain-rate regularization
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
from geometry import gaussian_curvature_brioschi
from losses import flatness_loss, elastic_energy_loss, strain_rate_loss


# =============================================================================
# Endpoint metrics
# =============================================================================

def metric_square(uv):
    """Identity metric: E=1, F=0, G=1."""
    N = uv.shape[0]
    E = torch.ones(N, device=uv.device, dtype=uv.dtype)
    F = torch.zeros(N, device=uv.device, dtype=uv.dtype)
    G = torch.ones(N, device=uv.device, dtype=uv.dtype)
    return (E, F, G)


def metric_circle(uv):
    """Metric of the elliptic map: (x,y) -> (x√(1-y²/2), y√(1-x²/2))."""
    x, y = uv[:, 0], uv[:, 1]
    x2, y2 = x * x, y * y

    su = torch.sqrt(1.0 - 0.5 * y2)
    sv = torch.sqrt(1.0 - 0.5 * x2)

    J11 = su
    J12 = x * (-y) / (2 * su + 1e-10)
    J21 = y * (-x) / (2 * sv + 1e-10)
    J22 = sv

    E = J11 * J11 + J21 * J21
    F = J11 * J12 + J21 * J22
    G = J12 * J12 + J22 * J22
    return (E, F, G)


# =============================================================================
# Sampling
# =============================================================================

def sample_collocation(n_points, lo=-0.85, hi=0.85, device='cpu'):
    """Sample (t, u, v) uniformly."""
    t = torch.rand(n_points, 1, device=device)
    uv = lo + (hi - lo) * torch.rand(n_points, 2, device=device)
    tuv = torch.cat([t, uv], dim=1)
    tuv.requires_grad_(True)
    return tuv


# =============================================================================
# Training
# =============================================================================

def train(
    n_steps=5000,
    n_collocation=2048,
    hidden_dim=128,
    n_layers=4,
    activation='silu',
    lr=1e-3,
    w_flat=1.0,
    w_elastic=0.0,
    w_strain_rate=0.0,
    margin=0.15,
    log_every=500,
    device='cpu',
):
    lo = -1.0 + margin
    hi = 1.0 - margin

    model = FundamentalFormNet(
        mode='2d',
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation=activation,
        endpoint_a_o=metric_square,
        endpoint_a_f=metric_circle,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    history = {
        'loss_total': [], 'loss_flat': [], 'loss_elastic': [],
        'loss_strain_rate': [], 'K_max': [], 'K_mean': [],
    }

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params} parameters, {activation} activation")
    print(f"Training: {n_steps} steps, {n_collocation} collocation points")
    print(f"Weights: flat={w_flat}, elastic={w_elastic}, strain_rate={w_strain_rate}")
    print()

    t_start = time.time()

    for step in range(n_steps):
        optimizer.zero_grad()

        tuv = sample_collocation(n_collocation, lo=lo, hi=hi, device=device)
        uv = tuv[:, 1:3]

        E, F, G = model(tuv)

        L_flat = flatness_loss(E, F, G, uv)

        L_elastic = torch.tensor(0.0, device=device)
        if w_elastic > 0:
            E0, F0, G0 = metric_square(uv)
            L_elastic = elastic_energy_loss(E, F, G, E0, F0, G0)

        L_strain_rate = torch.tensor(0.0, device=device)
        if w_strain_rate > 0:
            L_strain_rate = strain_rate_loss(E, F, G, tuv)

        loss = w_flat * L_flat + w_elastic * L_elastic + w_strain_rate * L_strain_rate

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            K = gaussian_curvature_brioschi(E, F, G, uv)
            K_max = torch.max(torch.abs(K)).item()
            K_mean = torch.mean(torch.abs(K)).item()

        history['loss_total'].append(loss.item())
        history['loss_flat'].append(L_flat.item())
        history['loss_elastic'].append(L_elastic.item())
        history['loss_strain_rate'].append(L_strain_rate.item())
        history['K_max'].append(K_max)
        history['K_mean'].append(K_mean)

        if (step + 1) % log_every == 0 or step == 0:
            elapsed = time.time() - t_start
            print(f"Step {step+1:5d}/{n_steps}  "
                  f"loss={loss.item():.4e}  "
                  f"|K|_max={K_max:.4e}  |K|_mean={K_mean:.4e}  "
                  f"({elapsed:.1f}s)")

    print(f"\nTraining complete in {time.time() - t_start:.1f}s")
    return model, history


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, n_t=11, K_grid=25, margin=0.15, device='cpu'):
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
        tuv.requires_grad_(True)

        E, F, G = model(tuv)

        # Curvature needs grad context
        K = gaussian_curvature_brioschi(E, F, G, tuv[:, 1:3])

        det_a = (E * G - F * F).detach()

        results.append({
            't': t_val.item(),
            'E': E.detach().reshape(K_grid, K_grid).cpu().numpy(),
            'F': F.detach().reshape(K_grid, K_grid).cpu().numpy(),
            'G': G.detach().reshape(K_grid, K_grid).cpu().numpy(),
            'K': K.detach().reshape(K_grid, K_grid).cpu().numpy(),
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
    axes[0].semilogy(history['loss_flat'], label='Flatness', color='C0', alpha=0.7)
    if max(history['loss_elastic']) > 0:
        axes[0].semilogy(history['loss_elastic'], label='Elastic', color='C1', alpha=0.7)
    if max(history['loss_strain_rate']) > 0:
        axes[0].semilogy(history['loss_strain_rate'], label='Strain rate', color='C2', alpha=0.7)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()

    axes[1].semilogy(history['K_max'], label='|K|_max', color='C0')
    axes[1].semilogy(history['K_mean'], label='|K|_mean', color='C1')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Curvature')
    axes[1].set_title('Gaussian Curvature During Training')
    axes[1].legend()

    t_vals = [r['t'] for r in results]
    K_max_t = [np.max(np.abs(r['K'])) for r in results]
    K_mean_t = [np.mean(np.abs(r['K'])) for r in results]
    axes[2].semilogy(t_vals, K_max_t, 'o-', label='|K|_max')
    axes[2].semilogy(t_vals, K_mean_t, 's-', label='|K|_mean')
    axes[2].set_xlabel('t')
    axes[2].set_ylabel('|K|')
    axes[2].set_title('Final Curvature vs Time')
    axes[2].legend()

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()

    # Metric fields
    n_t = len(results)
    indices = sorted(set([0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]))

    fig, axes = plt.subplots(3, len(indices), figsize=(4 * len(indices), 10))
    for col, idx in enumerate(indices):
        r = results[idx]
        for row, (field, name) in enumerate([
            (r['E'], 'E'), (r['F'], 'F'), (r['G'], 'G')
        ]):
            im = axes[row, col].imshow(field, origin='lower', cmap='viridis')
            axes[row, col].set_title(f"{name}, t={r['t']:.2f}")
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)

    plt.suptitle('Metric Components Along Trajectory')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'metric_fields.png'), dpi=150)
    plt.close()

    # Curvature fields
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 3.5))
    for col, idx in enumerate(indices):
        r = results[idx]
        vmax = max(np.max(np.abs(r['K'])), 1e-8)
        im = axes[col].imshow(r['K'], origin='lower', cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax)
        axes[col].set_title(f"K, t={r['t']:.2f}\nmax={np.max(np.abs(r['K'])):.2e}")
        plt.colorbar(im, ax=axes[col], fraction=0.046)

    plt.suptitle('Gaussian Curvature Along Trajectory')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'curvature_fields.png'), dpi=150)
    plt.close()

    # Determinant fields
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 3.5))
    for col, idx in enumerate(indices):
        r = results[idx]
        im = axes[col].imshow(r['det'], origin='lower', cmap='viridis')
        axes[col].set_title(
            f"det(a), t={r['t']:.2f}\n[{r['det'].min():.3f}, {r['det'].max():.3f}]")
        plt.colorbar(im, ax=axes[col], fraction=0.046)

    plt.suptitle('Metric Determinant Along Trajectory')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'determinant_fields.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {save_dir}/")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    save_dir = 'results_flat_trajectory'

    model, history = train(
        n_steps=5000,
        n_collocation=2048,
        hidden_dim=128,
        n_layers=4,
        activation='silu',
        lr=1e-3,
        w_flat=1.0,
        w_elastic=0.0,
        w_strain_rate=0.01,
        margin=0.15,
        log_every=500,
        device=device,
    )

    print("\nEvaluating on grid...")
    results = evaluate(model, n_t=11, K_grid=25, margin=0.15, device=device)

    for r in results:
        print(f"  t={r['t']:.2f}: |K|_max={np.max(np.abs(r['K'])):.4e}  "
              f"det=[{r['det'].min():.4f}, {r['det'].max():.4f}]")

    print("\nGenerating plots...")
    plot_results(results, history, save_dir=save_dir)

    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    print(f"Model saved to {save_dir}/model.pt")
