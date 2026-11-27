"""
Module 5: EPQD – Enhanced Parameterized Quantum Diffusion
===========================================================
Key optimisation vs naive implementation
-----------------------------------------
  PROBLEM: looping `for i in range(B): circuit(angles[i], weights)` causes
           O(B × NP) sequential quantum-circuit calls per batch → millions
           of calls per training run → hours of wall time.

  SOLUTION: PennyLane's default.qubit device supports *batch broadcasting*
            natively.  When `angles` has shape (B, n_qubits) the QNode
            vectorises over the batch axis automatically, costing only ONE
            circuit evaluation (with vectorised NumPy backend) per forward
            pass.  This is 100-300× faster than Python for-loops.

  Additional speed-ups:
    • diff_method="backprop"  – pure PyTorch autodiff, no parameter-shift
    • Quantum training set capped at MAX_Q_PATCHES patches
    • Classical pre/post MLPs are trained on the full dataset first (warm-up)
      so quantum weights only need fine-tuning
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module1_autoencoder import (ConvAutoencoder, get_dataloaders,
                                  DEVICE, LATENT_CHANNELS)
from module2_quantum_patch_encoding import (extract_patches,
                                             PATCH_SIZE, STRIDE, N_QUBITS)
from module4_classical_baselines import (FullLatentDataset, MLPDenoiser,
                                          UNetDenoiser, NOISE_STDDEV,
                                          PATCH_DIM, BATCH_SIZE, EPOCHS,
                                          LATENT_HW)
from module3_stitching_cnn import patches_to_latent


# ─────────────────────────── Proper Spatial Fold-back ────────────────────────
def fold_patches_spatial(patches: torch.Tensor,
                          n_h: int, n_w: int,
                          patch_size: int = PATCH_SIZE,
                          stride: int     = STRIDE,
                          n_channels: int = LATENT_CHANNELS) -> torch.Tensor:
    """
    Correct overlap-add fold-back that preserves spatial structure.

    patches : (B, NP, C*P*P)  — each patch IS a (C,P,P) spatial block
    Returns : (B, C, H, W)    — full latent map

    Previously this was done with .mean(-1) which collapsed the P×P spatial
    positions into a single value per channel — causing the gray outputs.
    This function places each (C,P,P) block back at its correct position.
    """
    B, NP, D = patches.shape
    P = patch_size
    C = n_channels
    H = (n_h - 1) * stride + P
    W = (n_w - 1) * stride + P

    # Reshape to spatial blocks: (B, NP, C, P, P)
    patches_sp = patches.view(B, NP, C, P, P)

    out   = torch.zeros(B, C, H, W, device=patches.device)
    count = torch.zeros(B, 1, H, W, device=patches.device)

    idx = 0
    for r in range(n_h):
        for c in range(n_w):
            rs = r * stride; re = rs + P
            cs = c * stride; ce = cs + P
            out[:, :, rs:re, cs:ce]   += patches_sp[:, idx]   # (B, C, P, P)
            count[:, :, rs:re, cs:ce] += 1.0
            idx += 1

    return out / count.clamp(min=1)   # (B, C, H, W)

# ─────────────────────────── Configuration ───────────────────────────────────
SAVE_DIR       = "outputs/module5"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("models",  exist_ok=True)

# Quantum configs to benchmark
QC_CONFIGS = [
    {"name": "EPQD-S",   "n_qubits": 4, "n_layers": 1, "ansatz": "basic"},
    {"name": "EPQD-M",   "n_qubits": 4, "n_layers": 2, "ansatz": "basic"},
    {"name": "EPQD-L",   "n_qubits": 4, "n_layers": 3, "ansatz": "basic"},
    {"name": "EPQD-IQP", "n_qubits": 4, "n_layers": 2, "ansatz": "iqp"},
]

WARMUP_EPOCHS  = 10      # classical pre-training epochs (fast, full dataset)
EPQD_EPOCHS    = 8       # quantum fine-tuning epochs (small dataset)
EPQD_LR        = 3e-3
WARMUP_LR      = 1e-3
EPQD_BATCH     = 128
MAX_Q_PATCHES  = 8_000   # cap quantum training set to keep runtime < 10 min


# ─────────────────────────── Batched QNode Factory ───────────────────────────
def _make_basic_circuit(n_qubits: int, n_layers: int):
    """
    Returns a QNode that accepts batched angles (B, n_qubits).
    PennyLane's default.qubit broadcasts over the batch dim automatically
    when diff_method="backprop", so there is NO Python for-loop over samples.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(angles, weights):
        # angles  : (B, n_qubits)  – batched input, broadcast by PennyLane
        # weights : (n_layers, n_qubits, 3) – shared across batch
        for i in range(n_qubits):
            qml.RY(angles[..., i], wires=i)      # broadcast-safe index
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            for i in range(n_qubits):
                qml.Rot(weights[layer, i, 0],
                        weights[layer, i, 1],
                        weights[layer, i, 2], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def _make_iqp_circuit(n_qubits: int, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(angles, weights):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(angles[..., i], wires=i)
        for i in range(n_qubits - 1):
            qml.IsingZZ(angles[..., 0] * angles[..., 1], wires=[i, i + 1])
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            for i in range(n_qubits):
                qml.Rot(weights[layer, i, 0],
                        weights[layer, i, 1],
                        weights[layer, i, 2], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def _make_reupload_xyz_circuit(n_qubits: int = 6, n_layers: int = 3):
    """
    Upgraded 6-qubit PQC with:
      1. Data Re-uploading: RY(angles) re-applied before every variational layer
      2. Multi-basis readout: PauliX, PauliY, PauliZ on every qubit -> 3*n_qubits features
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(angles, weights):
        for layer in range(n_layers):
            # Data re-uploading
            for i in range(n_qubits):
                qml.RY(angles[..., i], wires=i)
            # Entangling ring
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            # Variational rotations
            for i in range(n_qubits):
                qml.Rot(weights[layer, i, 0],
                        weights[layer, i, 1],
                        weights[layer, i, 2], wires=i)
        obs = (
            [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
            + [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
            + [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        )
        return obs

    return circuit


# ─────────────────────────── EPQD Model ──────────────────────────────────────
class EPQDDenoiser(nn.Module):
    """
    Enhanced Parameterized Quantum Diffusion denoiser.

    Forward pass (batched – NO Python loop over samples):
        (B, patch_dim) → pre_proj → (B, n_qubits) angles
                       → PQC [vectorised batch] → (B, n_qubits) ⟨Z⟩
                       → post_proj → (B, patch_dim)
        + residual skip
    """
    def __init__(self, name: str, n_qubits: int = 4, n_layers: int = 2,
                 ansatz: str = "basic", patch_dim: int = PATCH_DIM):
        super().__init__()
        self.name     = name
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical pre-projection → n_qubits angles
        self.pre_proj = nn.Sequential(
            nn.Linear(patch_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, n_qubits),
            nn.Tanh(),
        )

        # Quantum weights  (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        # Build batched QNode
        if ansatz == "iqp":
            self.circuit = _make_iqp_circuit(n_qubits, n_layers)
        else:
            self.circuit = _make_basic_circuit(n_qubits, n_layers)

        # Classical post-projection → patch_dim
        self.post_proj = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, patch_dim),
        )

        # Residual skip
        self.residual = nn.Linear(patch_dim, patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, patch_dim)
        No Python for-loop — PennyLane broadcasts over B automatically.
        """
        angles = self.pre_proj(x)                        # (B, n_qubits)
        angles = (angles + 1.0) * (torch.pi / 2.0)      # → [0, π]

        # Batched quantum forward  (B, n_qubits) → list of B-length tensors
        q_results = self.circuit(angles, self.q_weights)  # list of n_qubits tensors each (B,)
        q_out = torch.stack(q_results, dim=1).float()     # (B, n_qubits)

        return self.post_proj(q_out) + self.residual(x)


# ─────────────────────────── Dataset ─────────────────────────────────────────
class LatentPatchDatasetEPQD(Dataset):
    """Noisy→clean patch pairs (flat, for patch-level denoising)."""
    def __init__(self, image_loader, ae_encoder,
                 noise_std: float = NOISE_STDDEV, max_batches: int = 30):
        super().__init__()
        self.clean_patches = []
        ae_encoder.eval()
        with torch.no_grad():
            for i, (imgs, _) in enumerate(image_loader):
                if i >= max_batches:
                    break
                z = ae_encoder(imgs.to(DEVICE)).cpu()
                patches, _, _ = extract_patches(z, PATCH_SIZE, STRIDE)
                self.clean_patches.append(patches.view(-1, patches.shape[-1]))
        self.clean_patches = torch.cat(self.clean_patches, dim=0)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.clean_patches)

    def __getitem__(self, idx):
        clean = self.clean_patches[idx]
        noisy = clean + torch.randn_like(clean) * self.noise_std
        return noisy, clean


# ─────────────────────────── Training ────────────────────────────────────────
def _build_dataloaders(ae_model, train_loader_imgs, test_loader_imgs):
    print("\n[Building patch datasets …]")
    train_ds = LatentPatchDatasetEPQD(train_loader_imgs, ae_model.encoder,
                                       max_batches=30)
    test_ds  = LatentPatchDatasetEPQD(test_loader_imgs,  ae_model.encoder,
                                       max_batches=8)
    print(f"  Full  train: {len(train_ds):,}  |  test: {len(test_ds):,}")

    # Cap quantum training set for speed
    q_size   = min(MAX_Q_PATCHES, len(train_ds))
    q_idx    = torch.randperm(len(train_ds))[:q_size]
    q_ds     = Subset(train_ds, q_idx.tolist())

    q_size_t  = min(MAX_Q_PATCHES // 4, len(test_ds))
    q_idx_t   = torch.randperm(len(test_ds))[:q_size_t]
    q_ds_t    = Subset(test_ds, q_idx_t.tolist())

    print(f"  Quantum subset: train={q_size:,}  |  test={q_size_t:,}")

    full_dl = DataLoader(train_ds, EPQD_BATCH, shuffle=True,  num_workers=0)
    full_te = DataLoader(test_ds,  EPQD_BATCH, shuffle=False, num_workers=0)
    q_dl    = DataLoader(q_ds,     EPQD_BATCH, shuffle=True,  num_workers=0)
    q_te    = DataLoader(q_ds_t,   EPQD_BATCH, shuffle=False, num_workers=0)

    return full_dl, full_te, q_dl, q_te


def _classical_warmup(model: EPQDDenoiser, full_dl, full_te,
                       epochs: int = WARMUP_EPOCHS):
    """
    Pre-train the CLASSICAL layers (pre_proj, post_proj, residual) only.
    Quantum weights are frozen during warmup.
    At this stage the quantum layer is bypassed: we route angles directly
    through a linear approximation so the classical heads learn a useful
    latent-space mapping before quantum fine-tuning.
    """
    # Freeze quantum weights
    model.q_weights.requires_grad_(False)

    # Build a temporary bypass: pre_proj → linear → post_proj  (no circuit)
    n_q = model.n_qubits
    bypass = nn.Linear(n_q, n_q).to(DEVICE)

    classical_params = (list(model.pre_proj.parameters()) +
                        list(model.post_proj.parameters()) +
                        list(model.residual.parameters()) +
                        list(bypass.parameters()))
    opt  = optim.Adam(classical_params, lr=WARMUP_LR)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    model.to(DEVICE); bypass.to(DEVICE)

    print(f"  [Warm-up {epochs} epochs – classical layers only, no quantum] …")
    for epoch in range(1, epochs + 1):
        model.train(); bypass.train()
        run = 0.0
        for noisy, clean in full_dl:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            angles  = model.pre_proj(noisy)
            angles  = (angles + 1.0) * (torch.pi / 2.0)
            q_approx = bypass(angles).float()
            pred    = model.post_proj(q_approx) + model.residual(noisy)
            loss    = crit(pred, clean)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * noisy.size(0)
        sch.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Warm-up epoch {epoch:2d}/{epochs} | "
                  f"Train MSE: {run/len(full_dl.dataset):.6f}")

    # Unfreeze quantum weights for fine-tuning
    model.q_weights.requires_grad_(True)
    del bypass


def train_epqd(model: EPQDDenoiser, full_dl, full_te, q_dl, q_te,
               warmup_epochs: int = WARMUP_EPOCHS,
               quantum_epochs: int = EPQD_EPOCHS,
               lr: float = EPQD_LR):

    crit = nn.MSELoss()
    model.to(DEVICE)

    # ── Phase 1: classical warm-up (fast, full dataset) ──
    _classical_warmup(model, full_dl, full_te, warmup_epochs)

    # ── Phase 2: quantum fine-tuning (small dataset) ──
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=quantum_epochs)
    train_losses, test_losses = [], []
    t0 = time.time()

    print(f"  [Quantum fine-tune {quantum_epochs} epochs, "
          f"dataset={len(q_dl.dataset):,}] …")
    for epoch in range(1, quantum_epochs + 1):
        model.train()
        run = 0.0
        for noisy, clean in q_dl:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            pred = model(noisy)
            loss = crit(pred, clean)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * noisy.size(0)
        train_losses.append(run / len(q_dl.dataset))

        model.eval()
        run = 0.0
        with torch.no_grad():
            for noisy, clean in q_te:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                pred = model(noisy)
                run += crit(pred, clean).item() * noisy.size(0)
        test_losses.append(run / len(q_te.dataset))
        sch.step()

        elapsed = time.time() - t0
        print(f"    [{model.name}] Q-epoch {epoch:2d}/{quantum_epochs} | "
              f"Train: {train_losses[-1]:.6f} | Test: {test_losses[-1]:.6f} | "
              f"{elapsed:.0f}s")

    return train_losses, test_losses


# ─────────────────────────── Reconstruction ──────────────────────────────────
@torch.no_grad()
def epqd_full_reconstruction(epqd_model, ae_model, imgs):
    """
    Full reconstruction pipeline:
      img → encoder → clean_z
                        ↓ +noise
                      noisy_z → extract patches
                                    ↓ EPQD denoise
                                denoised_patches
                                    ↓ spatial fold-back (C,P,P) blocks
                                denoised_z → decoder → image
    """
    imgs    = imgs.to(DEVICE)
    clean_z = ae_model.encoder(imgs)
    noisy_z = clean_z + torch.randn_like(clean_z) * NOISE_STDDEV

    patches, n_h, n_w = extract_patches(noisy_z.cpu(), PATCH_SIZE, STRIDE)
    patches = patches.to(DEVICE)

    epqd_model.eval()
    B, NP, D = patches.shape
    denoised_flat = epqd_model(patches.view(B * NP, D))   # (B*NP, C*P*P)
    denoised_p    = denoised_flat.view(B, NP, D)

    # ── Proper spatial fold-back (not mean!) ──
    refined_z = fold_patches_spatial(
        denoised_p.cpu(), n_h, n_w, PATCH_SIZE, STRIDE, LATENT_CHANNELS
    ).to(DEVICE)

    return ae_model.decoder(clean_z), ae_model.decoder(noisy_z), ae_model.decoder(refined_z)


# ─────────────────────────── Visualisation ───────────────────────────────────
def save_epqd_reconstruction_grid(best_epqd, ae_model, test_loader,
                                   model_name="EPQD-Best", n=10):
    imgs, _ = next(iter(test_loader))
    imgs = imgs[:n]
    clean_r, noisy_r, epqd_r = epqd_full_reconstruction(best_epqd, ae_model, imgs)

    rows   = [imgs.cpu(), noisy_r.cpu(), epqd_r.cpu()]
    titles = ["Original", "Noisy Latent Recon",
              f"EPQD Denoised ({model_name})"]

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    for r, (rim, title) in enumerate(zip(rows, titles)):
        rn = rim.numpy()
        for c in range(n):
            axes[r, c].imshow(rn[c, 0] * 0.5 + 0.5, cmap="gray",
                              vmin=0, vmax=1, interpolation="none")
            axes[r, c].axis("off")
        axes[r, 0].set_title(title, fontsize=10, loc="left")

    plt.suptitle(f"Module 5 – EPQD ({model_name})", fontsize=13)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, f"epqd_recon_{model_name}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(); print(f"  Saved → {out}")
    return out


def save_all_model_comparison(all_models_info, ae_model, test_loader, n=8):
    imgs, _ = next(iter(test_loader))
    imgs = imgs[:n]
    ae_model.eval()

    with torch.no_grad():
        clean_z  = ae_model.encoder(imgs.to(DEVICE))
        noisy_z  = clean_z + torch.randn_like(clean_z) * NOISE_STDDEV
        clean_r  = ae_model.decoder(clean_z).cpu().numpy()   # AE verification row
        noisy_r  = ae_model.decoder(noisy_z).cpu().numpy()

    # ── MLP row (patch-based, proper fold-back) ──
    mlp = all_models_info.get("mlp")
    if mlp:
        mlp.eval()
        with torch.no_grad():
            p, n_h, n_w = extract_patches(noisy_z.cpu(), PATCH_SIZE, STRIDE)
            p  = p.to(DEVICE)
            B, NP, D = p.shape
            mp = mlp(p.view(B * NP, D)).view(B, NP, D)        # (B, NP, C*P*P)
            # Proper spatial fold-back
            mlp_z = fold_patches_spatial(mp.cpu(), n_h, n_w,
                                          PATCH_SIZE, STRIDE, LATENT_CHANNELS).to(DEVICE)
            mlp_r = ae_model.decoder(mlp_z).cpu().numpy()
    else:
        mlp_r = noisy_r

    # ── U-Net row (full latent, no patches needed) ──
    unet = all_models_info.get("unet")
    if unet:
        unet.eval()
        with torch.no_grad():
            unet_r = ae_model.decoder(unet(noisy_z)).cpu().numpy()
    else:
        unet_r = noisy_r

    # ── EPQD rows (patch-based, proper fold-back via epqd_full_reconstruction) ──
    epqd_rows = []
    for cname, em in all_models_info.get("epqd", {}).items():
        _, _, er = epqd_full_reconstruction(em, ae_model, imgs)
        epqd_rows.append((cname, er.cpu().numpy()))

    # ── Build figure ──
    # Rows: Original | AE Clean Recon | Noisy Recon | MLP | U-Net | EPQD×N
    all_rows   = [imgs.numpy(), clean_r, noisy_r, mlp_r, unet_r]
    all_titles = ["Original",
                  "Conv-AE Clean Recon (verify AE)",
                  f"Noisy Latent Recon (σ={NOISE_STDDEV})",
                  "MLP Denoised",
                  "U-Net Denoised"]
    for cname, er in epqd_rows:
        all_rows.append(er); all_titles.append(f"EPQD ({cname})")

    n_rows = len(all_rows)
    fig, axes = plt.subplots(n_rows, n, figsize=(2 * n, 2 * n_rows))

    for r, (rim, title) in enumerate(zip(all_rows, all_titles)):
        for c in range(n):
            axes[r, c].imshow(rim[c, 0] * 0.5 + 0.5, cmap="gray",
                              vmin=0, vmax=1, interpolation="none")
            axes[r, c].axis("off")
        axes[r, 0].set_title(title, fontsize=9, loc="left", fontweight="bold")

    plt.suptitle("PHQD – Full Model Comparison (Module 5)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "PHQD_full_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(); print(f"  Saved → {out}")
    return out


def save_mse_chart(mse_results: dict):
    names  = list(mse_results.keys())
    values = list(mse_results.values())
    colors = ["steelblue"     if "MLP"  in n else
              "darkorange"    if "UNet" in n or "U-Net" in n else
              "mediumseagreen" for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.6), 5))
    bars = ax.bar(names, values, color=colors, edgecolor="k", linewidth=0.5)
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=9)
    ax.set_ylabel("Test MSE (patch denoising)")
    ax.set_title("PHQD – MSE: Classical vs Quantum Denoisers",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "PHQD_MSE_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(); print(f"  Saved → {out}")

    print("\n" + "=" * 50)
    print(f"  {'Model':<20} {'Test MSE':>10}")
    print("-" * 50)
    for name, mse in sorted(mse_results.items(), key=lambda x: x[1]):
        marker = "  ◀ BEST" if mse == min(values) else ""
        print(f"  {name:<20} {mse:10.6f}{marker}")
    print("=" * 50)


def save_quantum_circuit_diagram(n_qubits=4, n_layers=2, weights_np=None, angles_demo=None, model_name="EPQD"):
    try:
        dev = qml.device("default.qubit", wires=n_qubits)
        if weights_np is None:
            weights_np = np.zeros((n_layers, n_qubits, 3))
        if angles_demo is None:
            angles_demo = np.zeros(n_qubits)

        @qml.qnode(dev)
        def circuit(angles, weights):
            for i in range(n_qubits):
                qml.RY(angles[i], wires=i)
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
                for i in range(n_qubits):
                    qml.Rot(weights[layer, i, 0],
                            weights[layer, i, 1],
                            weights[layer, i, 2], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        fig, ax = qml.draw_mpl(circuit, decimals=2)(angles_demo, weights_np)
        ax.set_title(f"{model_name} PQC – {n_qubits} qubits, {n_layers} layers")
        out = os.path.join(SAVE_DIR, "pqc_circuit_diagram.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig); print(f"  Saved → {out}")
    except Exception as e:
        print(f"  Circuit diagram skipped: {e}")


# ─────────────────────────── Entry Point ─────────────────────────────────────
def main():
    print("=" * 60)
    print("MODULE 5 – EPQD: Enhanced Parameterized Quantum Diffusion")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Strategy: classical warm-up ({WARMUP_EPOCHS} ep) "
          f"→ quantum fine-tune ({EPQD_EPOCHS} ep, ≤{MAX_Q_PATCHES:,} patches)")
    print(f"  Batched QNode (no Python for-loop) → ~100-300× faster")

    train_loader, test_loader = get_dataloaders(EPQD_BATCH)

    # Load pre-trained Conv-AE
    ae_model = ConvAutoencoder().to(DEVICE)
    if os.path.exists("models/conv_ae.pth"):
        ae_model.load_state_dict(torch.load("models/conv_ae.pth",
                                             map_location=DEVICE))
        print("  Loaded Conv-AE ✓")
    ae_model.eval()

    full_dl, full_te, q_dl, q_te = _build_dataloaders(
        ae_model, train_loader, test_loader)

    # Circuit diagram
    print("\n[Drawing PQC circuit diagram …]")
    save_quantum_circuit_diagram()

    # ── Train all EPQD configs ──
    mse_results = {}
    best_model, best_mse, best_name = None, float("inf"), ""
    epqd_models, all_losses = {}, {}

    for cfg in QC_CONFIGS:
        print(f"\n{'─'*60}")
        print(f"[{cfg['name']} | qubits={cfg['n_qubits']} "
              f"layers={cfg['n_layers']} ansatz={cfg['ansatz']}]")
        model = EPQDDenoiser(
            name=cfg["name"], n_qubits=cfg["n_qubits"],
            n_layers=cfg["n_layers"], ansatz=cfg["ansatz"],
            patch_dim=PATCH_DIM,
        )
        tr_l, te_l = train_epqd(model, full_dl, full_te, q_dl, q_te,
                                  WARMUP_EPOCHS, EPQD_EPOCHS, EPQD_LR)
        mse = te_l[-1]
        mse_results[cfg["name"]] = mse
        all_losses[cfg["name"]]  = (tr_l, te_l)
        epqd_models[cfg["name"]] = model

        torch.save(model.state_dict(), f"models/epqd_{cfg['name']}.pth")
        print(f"  Saved → models/epqd_{cfg['name']}.pth | Test MSE: {mse:.6f}")

        if mse < best_mse:
            best_mse, best_model, best_name = mse, model, cfg["name"]

    print(f"\n  Best EPQD: {best_name}  (MSE={best_mse:.6f})")

    # ── Load classical baselines for joint comparison ──
    all_models_info = {"epqd": epqd_models}
    crit = nn.MSELoss()

    mlp = MLPDenoiser(PATCH_DIM).to(DEVICE)
    if os.path.exists("models/mlp_denoiser.pth"):
        mlp.load_state_dict(torch.load("models/mlp_denoiser.pth",
                                        map_location=DEVICE))
        all_models_info["mlp"] = mlp; mlp.eval()
        run = 0.0
        with torch.no_grad():
            for noisy, clean in q_te:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                run += crit(mlp(noisy), clean).item() * noisy.size(0)
        mse_results["MLP"] = run / len(q_te.dataset)
        print(f"  MLP baseline MSE : {mse_results['MLP']:.6f}")

    unet = UNetDenoiser().to(DEVICE)
    if os.path.exists("models/unet_denoiser.pth"):
        unet.load_state_dict(torch.load("models/unet_denoiser.pth",
                                         map_location=DEVICE))
        all_models_info["unet"] = unet; unet.eval()
        from module4_classical_baselines import FullLatentDataset
        full_ds = FullLatentDataset(test_loader, ae_model.encoder,
                                     max_batches=8)
        full_dl2 = DataLoader(full_ds, 64, False, num_workers=0)
        run = 0.0
        with torch.no_grad():
            for noisy, clean in full_dl2:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                run += crit(unet(noisy), clean).item() * noisy.size(0)
        mse_results["U-Net"] = run / len(full_dl2.dataset)
        print(f"  U-Net baseline MSE: {mse_results['U-Net']:.6f}")

    # ── Outputs ──
    print("\n[Saving outputs …]")
    save_mse_chart(mse_results)
    save_epqd_reconstruction_grid(best_model, ae_model, test_loader, best_name)
    for cname, em in epqd_models.items():
        save_epqd_reconstruction_grid(em, ae_model, test_loader, cname)
    save_all_model_comparison(all_models_info, ae_model, test_loader)

    # Loss curves
    fig, axes = plt.subplots(1, len(QC_CONFIGS),
                              figsize=(5 * len(QC_CONFIGS), 4))
    if len(QC_CONFIGS) == 1: axes = [axes]
    for ax, cfg in zip(axes, QC_CONFIGS):
        tr_l, te_l = all_losses[cfg["name"]]
        ax.plot(tr_l, label="Train"); ax.plot(te_l, label="Test")
        ax.set_title(cfg["name"]); ax.set_xlabel("Q-epoch")
        ax.set_ylabel("MSE"); ax.legend(); ax.grid(True)
    plt.suptitle("Module 5 – EPQD Quantum Fine-tune Curves", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "epqd_loss_curves.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    print("\nMODULE 5 (EPQD) COMPLETE ✓")
    return best_model, mse_results


# ─────────────────────────── Upgraded EPQD-V2 (6q + X/Y/Z + Re-upload + Joint) ───────
class EPQDV2Denoiser(nn.Module):
    """
    Upgraded 6-qubit EPQD denoiser with:
      - 6 qubits + 3 data-reuploading layers
      - Multi-basis X/Y/Z measurements (18 quantum features)
      - Wider pre/post classical projections + residual skip
    """
    def __init__(self, name: str = "EPQD-6Q-V2", n_qubits: int = 6,
                 n_layers: int = 3, patch_dim: int = PATCH_DIM):
        super().__init__()
        self.name = name
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.q_out_dim = 3 * n_qubits

        self.pre_proj = nn.Sequential(
            nn.Linear(patch_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, n_qubits),
            nn.Tanh(),
        )
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )
        self.circuit = _make_reupload_xyz_circuit(n_qubits, n_layers)

        self.post_proj = nn.Sequential(
            nn.Linear(self.q_out_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, patch_dim),
        )
        self.residual = nn.Linear(patch_dim, patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = (self.pre_proj(x) + 1.0) * (torch.pi / 2.0)
        q_results = self.circuit(angles, self.q_weights)
        q_out = torch.stack(q_results, dim=1).float()   # (B, 18)
        return self.post_proj(q_out) + self.residual(x)


def _ssim_01(pred_01: torch.Tensor, target_01: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Differentiable SSIM over (B, 1, H, W) in [0, 1]."""
    import torch.nn.functional as F
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    coords = torch.arange(window_size, dtype=torch.float32, device=pred_01.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    pad = window_size // 2
    mu1 = F.conv2d(pred_01, window, padding=pad)
    mu2 = F.conv2d(target_01, window, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1_sq = F.conv2d(pred_01 * pred_01, window, padding=pad) - mu1_sq
    s2_sq = F.conv2d(target_01 * target_01, window, padding=pad) - mu2_sq
    s12 = F.conv2d(pred_01 * target_01, window, padding=pad) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2))
    return ssim_map.mean()


def run_v2_benchmark():
    """
    Train and benchmark EPQD-6Q-V2 (6 qubits + X/Y/Z + data re-uploading +
    joint StitchingCNN + image/SSIM loss) against MLP, U-Net, and EPQD-L.
    """
    import torch.nn.functional as F
    from module3_stitching_cnn import StitchingCNN

    print("=" * 68)
    print("MODULE 5 (V2) – EPQD-6Q-V2 (6q + X/Y/Z + Re-uploading + Joint Stitch)")
    print("=" * 68)

    train_loader, test_loader = get_dataloaders(EPQD_BATCH)
    ae_model = ConvAutoencoder().to(DEVICE)
    ae_model.load_state_dict(torch.load("models/conv_ae.pth", map_location=DEVICE))
    ae_model.eval()
    for p in ae_model.parameters():
        p.requires_grad_(False)

    full_dl, full_te, q_dl, q_te = _build_dataloaders(ae_model, train_loader, test_loader)
    model = EPQDV2Denoiser("EPQD-6Q-V2", n_qubits=6, n_layers=3, patch_dim=PATCH_DIM).to(DEVICE)

    # ── Phase 1: Classical warm-up (15 epochs on full 138K patches) ──
    model.q_weights.requires_grad_(False)
    bypass = nn.Linear(model.n_qubits, model.q_out_dim).to(DEVICE)
    c_params = (list(model.pre_proj.parameters()) +
                list(model.post_proj.parameters()) +
                list(model.residual.parameters()) +
                list(bypass.parameters()))
    opt1 = optim.Adam(c_params, lr=1e-3)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=15)
    crit = nn.MSELoss()

    print("\n[Phase 1: Classical warm-up (15 epochs, full patch dataset)]")
    for ep in range(1, 16):
        model.train(); bypass.train()
        run = 0.0
        for noisy, clean in full_dl:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            ang = (model.pre_proj(noisy) + 1.0) * (torch.pi / 2.0)
            pred = model.post_proj(bypass(ang).float()) + model.residual(noisy)
            loss = crit(pred, clean)
            opt1.zero_grad(); loss.backward(); opt1.step()
            run += loss.item() * noisy.size(0)
        sch1.step()
        if ep in (1, 5, 10, 15):
            print(f"  Warm-up {ep:2d}/15 | Train MSE: {run/len(full_dl.dataset):.6f}")
    model.q_weights.requires_grad_(True)
    del bypass

    # ── Phase 2: Quantum patch fine-tuning (8 epochs) ──
    opt2 = optim.Adam(model.parameters(), lr=2e-3)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=8)
    print("\n[Phase 2: 6-Qubit X/Y/Z Data-Reuploading Fine-tune (8 epochs)]")
    t0 = time.time()
    for ep in range(1, 9):
        model.train()
        run = 0.0
        for noisy, clean in q_dl:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            loss = crit(model(noisy), clean)
            opt2.zero_grad(); loss.backward(); opt2.step()
            run += loss.item() * noisy.size(0)
        model.eval()
        te_run = 0.0
        with torch.no_grad():
            for noisy, clean in q_te:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                te_run += crit(model(noisy), clean).item() * noisy.size(0)
        sch2.step()
        print(f"  Q-epoch {ep:2d}/8 | Train: {run/len(q_dl.dataset):.6f} | "
              f"Test: {te_run/len(q_te.dataset):.6f} | {time.time()-t0:.0f}s")

    # ── Phase 3: Joint EPQD-6Q-V2 + StitchingCNN + Image/SSIM Fine-tuning ──
    stitch_v2 = StitchingCNN().to(DEVICE)
    if os.path.exists("models/stitching_cnn.pth"):
        stitch_v2.load_state_dict(torch.load("models/stitching_cnn.pth", map_location=DEVICE))

    joint_loader, _ = get_dataloaders(16)
    opt3 = optim.Adam(list(model.parameters()) + list(stitch_v2.parameters()), lr=5e-4)
    sch3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=6)

    print("\n[Phase 3: Joint EPQD-6Q-V2 + StitchingCNN + Image/SSIM Fine-tuning (6 epochs)]")
    max_joint_batches = 60  # 960 images/epoch
    for ep in range(1, 7):
        model.train(); stitch_v2.train()
        run_loss = 0.0
        n_imgs = 0
        for b_idx, (imgs, _) in enumerate(joint_loader):
            if b_idx >= max_joint_batches:
                break
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                clean_z = ae_model.encoder(imgs)
            noisy_z = clean_z + torch.randn_like(clean_z) * NOISE_STDDEV
            patches, n_h, n_w = extract_patches(noisy_z.cpu(), PATCH_SIZE, STRIDE)
            patches = patches.to(DEVICE)
            B, NP, D = patches.shape
            den_p = model(patches.view(B * NP, D)).view(B, NP, D)
            folded_z = fold_patches_spatial(den_p, n_h, n_w, PATCH_SIZE, STRIDE, LATENT_CHANNELS)
            refined_z = stitch_v2(folded_z)
            recon_img = ae_model.decoder(refined_z)

            r01 = (recon_img * 0.5 + 0.5).clamp(0.0, 1.0)
            t01 = (imgs * 0.5 + 0.5).clamp(0.0, 1.0)
            loss = (
                F.mse_loss(refined_z, clean_z)
                + 0.5 * F.mse_loss(r01, t01)
                + 0.2 * (1.0 - _ssim_01(r01, t01))
            )
            opt3.zero_grad(); loss.backward(); opt3.step()
            run_loss += loss.item() * B
            n_imgs += B
        sch3.step()
        print(f"  Joint epoch {ep:2d}/6 | Loss: {run_loss/n_imgs:.6f} | {time.time()-t0:.0f}s")

    torch.save(model.state_dict(), "models/epqd_EPQD-6Q-V2.pth")
    torch.save(stitch_v2.state_dict(), "models/stitching_cnn_v2.pth")
    print("\n  Saved → models/epqd_EPQD-6Q-V2.pth & models/stitching_cnn_v2.pth")

    # ── Benchmark against MLP, U-Net, EPQD-L on 64 test images ──
    _, eval_loader = get_dataloaders(64)
    eval_imgs, _ = next(iter(eval_loader))
    eval_imgs = eval_imgs.to(DEVICE)
    o01 = (eval_imgs * 0.5 + 0.5).clamp(0.0, 1.0)

    torch.manual_seed(42)
    with torch.no_grad():
        clean_z = ae_model.encoder(eval_imgs)
        noisy_z = clean_z + torch.randn_like(clean_z) * NOISE_STDDEV
        p_eval, nh_e, nw_e = extract_patches(noisy_z, PATCH_SIZE, STRIDE)
        B, NP, D = p_eval.shape

        mlp = MLPDenoiser(PATCH_DIM).to(DEVICE)
        mlp.load_state_dict(torch.load("models/mlp_denoiser.pth", map_location=DEVICE))
        mlp.eval()
        mlp_z = fold_patches_spatial(mlp(p_eval.view(B * NP, D)).view(B, NP, D),
                                      nh_e, nw_e, PATCH_SIZE, STRIDE, LATENT_CHANNELS)
        mlp_r = ae_model.decoder(mlp_z)

        unet = UNetDenoiser().to(DEVICE)
        unet.load_state_dict(torch.load("models/unet_denoiser.pth", map_location=DEVICE))
        unet.eval()
        unet_z = unet(noisy_z)
        unet_r = ae_model.decoder(unet_z)

        epqd_l = EPQDDenoiser("EPQD-L", n_qubits=4, n_layers=3, ansatz="basic", patch_dim=PATCH_DIM).to(DEVICE)
        epqd_l.load_state_dict(torch.load("models/epqd_EPQD-L.pth", map_location=DEVICE))
        epqd_l.eval()
        epqd_l_z = fold_patches_spatial(epqd_l(p_eval.view(B * NP, D)).view(B, NP, D),
                                         nh_e, nw_e, PATCH_SIZE, STRIDE, LATENT_CHANNELS)
        epqd_l_r = ae_model.decoder(epqd_l_z)

        model.eval(); stitch_v2.eval()
        v2_folded_z = fold_patches_spatial(model(p_eval.view(B * NP, D)).view(B, NP, D),
                                           nh_e, nw_e, PATCH_SIZE, STRIDE, LATENT_CHANNELS)
        v2_z = stitch_v2(v2_folded_z)
        v2_r = ae_model.decoder(v2_z)

        noisy_r = ae_model.decoder(noisy_z)

    def _eval_row(name, z_pred, img_pred):
        r01 = (img_pred * 0.5 + 0.5).clamp(0.0, 1.0)
        lat_mse = F.mse_loss(z_pred, clean_z).item()
        img_mse = F.mse_loss(r01, o01).item()
        rmse = img_mse ** 0.5
        mae = F.l1_loss(r01, o01).item()
        psnr = 10.0 * np.log10(1.0 / max(img_mse, 1e-12))
        ssim = _ssim_01(r01, o01).item()
        cos = F.cosine_similarity(r01.flatten(1), o01.flatten(1)).mean().item()
        return (name, lat_mse, img_mse, rmse, mae, psnr, ssim, cos)

    rows = [
        _eval_row("Noisy (s=0.3)", noisy_z, noisy_r),
        _eval_row("U-Net (Classical)", unet_z, unet_r),
        _eval_row("MLP (Classical)", mlp_z, mlp_r),
        _eval_row("EPQD-L (4q V1)", epqd_l_z, epqd_l_r),
        _eval_row("EPQD-6Q-V2 (Ours)", v2_z, v2_r),
    ]

    print("\n" + "=" * 92)
    print(f"  {'Model':<20} {'LatMSE':>9} {'ImgMSE':>9} {'RMSE':>8} {'MAE':>8} {'PSNR':>8} {'SSIM':>7} {'CosSim':>8}")
    print("-" * 92)
    for nm, lm, im, rm, ma, ps, ss, cs in rows:
        print(f"  {nm:<20} {lm:9.6f} {im:9.6f} {rm:8.5f} {ma:8.5f} {ps:7.2f}dB {ss:7.4f} {cs:8.4f}")
    print("=" * 92)

    # Save visual comparison grid
    n_show = 8
    vis_rows = [eval_imgs[:n_show].cpu().numpy(),
                noisy_r[:n_show].cpu().numpy(),
                unet_r[:n_show].cpu().numpy(),
                mlp_r[:n_show].cpu().numpy(),
                epqd_l_r[:n_show].cpu().numpy(),
                v2_r[:n_show].cpu().numpy()]
    vis_titles = ["Original", "Noisy (s=0.3)", "U-Net", "MLP", "EPQD-L (4q V1)", "EPQD-6Q-V2 (6q+XYZ+Joint)"]
    fig, axes = plt.subplots(len(vis_rows), n_show, figsize=(2 * n_show, 2 * len(vis_rows)))
    for r_idx, (r_imgs, title) in enumerate(zip(vis_rows, vis_titles)):
        for c_idx in range(n_show):
            axes[r_idx, c_idx].imshow(r_imgs[c_idx, 0] * 0.5 + 0.5, cmap="gray", vmin=0, vmax=1, interpolation="none")
            axes[r_idx, c_idx].axis("off")
        axes[r_idx, 0].set_title(title, fontsize=9, loc="left", fontweight="bold")
    plt.suptitle("PHQD-V2 Comparison: Upgraded 6-Qubit EPQD vs Classical Baselines", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "PHQD_V2_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {os.path.join(SAVE_DIR, 'PHQD_V2_comparison.png')}")

    return rows


if __name__ == "__main__":
    import sys
    if "--v2" in sys.argv:
        run_v2_benchmark()
    else:
        main()
