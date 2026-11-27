"""
Module 3: Classical Stitching CNN  (PURELY CLASSICAL – no quantum in training)
================================================================================
Responsibility:
  • Take overlapping patches of a latent map (from patch extraction)
  • Fold them back (overlap-add) into a spatial feature map
  • Refine the stitched map with a small residual CNN
  • Output a clean 64×7×7 latent ready for the Conv-AE decoder

Training signal: the CNN learns to denoise / refine slightly perturbed latent
maps so that the decoder still produces sharp images.  No quantum circuit is
called during training – that is Module 2's job.

Runtime: typically < 2 minutes on CPU for 15 epochs.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module1_autoencoder import (ConvAutoencoder, get_dataloaders,
                                  LATENT_CHANNELS, LATENT_DIM, DEVICE)
from module2_quantum_patch_encoding import (extract_patches,
                                             PATCH_SIZE, STRIDE, N_QUBITS)

# ─────────────────────────── Configuration ───────────────────────────────────
LATENT_HW      = 7
EPOCHS_STITCH  = 15
LR_STITCH      = 1e-3
BATCH_SIZE     = 128
NOISE_STD      = 0.15      # small perturbation used as training signal
SAVE_DIR       = "outputs/module3"
MODEL_PATH     = "models/stitching_cnn.pth"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("models",  exist_ok=True)


# ─────────────────────────── Patch Fold-Back (overlap-add) ───────────────────
def patches_to_latent(patch_features: torch.Tensor,
                      n_h: int, n_w: int,
                      patch_size: int = PATCH_SIZE,
                      stride: int     = STRIDE,
                      out_channels: int = N_QUBITS) -> torch.Tensor:
    """
    Fold processed patches back into a spatial feature map using
    overlap-add with count normalisation.

    patch_features : (B, n_h*n_w, out_channels)
    Returns        : (B, out_channels, H, W)
    """
    B, NP, C = patch_features.shape
    H = (n_h - 1) * stride + patch_size
    W = (n_w - 1) * stride + patch_size

    out   = torch.zeros(B, C, H, W, device=patch_features.device)
    count = torch.zeros(B, 1, H, W, device=patch_features.device)

    idx = 0
    for r in range(n_h):
        for c in range(n_w):
            rs = r * stride; re = rs + patch_size
            cs = c * stride; ce = cs + patch_size
            feat = patch_features[:, idx, :].view(B, C, 1, 1)
            out[:, :, rs:re, cs:ce]   += feat.expand(B, C, patch_size, patch_size)
            count[:, :, rs:re, cs:ce] += 1.0
            idx += 1

    return out / count.clamp(min=1)   # (B, C, H, W)


# ─────────────────────────── Stitching Refinement CNN ────────────────────────
class ResBlock(nn.Module):
    """Single residual block: 3×3 conv → BN → GELU → 3×3 conv → BN + skip."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class StitchingCNN(nn.Module):
    """
    Pure classical CNN that refines a spatial latent map.

    Input  : (B, in_channels, H, W)   – e.g. (B, 64, 7, 7) noisy latent
    Output : (B, out_channels, H, W)  – refined latent (same spatial size)

    Architecture:
        1×1 expand → 3 ResBlocks → 1×1 project → Tanh activation
        + 1×1 skip connection
    """
    def __init__(self,
                 in_channels : int = LATENT_CHANNELS,
                 out_channels: int = LATENT_CHANNELS,
                 mid_channels: int = 64):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(mid_channels),
            ResBlock(mid_channels),
            ResBlock(mid_channels),
        )
        self.head = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.Tanh(),
        )
        # 1×1 skip: maps in → out if dims differ
        self.skip = (nn.Conv2d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.res_blocks(self.stem(x))) + self.skip(x)


# ─────────────────────────── Dataset (purely classical) ──────────────────────
class LatentRefinementDataset(Dataset):
    """
    Generates (noisy_latent, clean_latent) pairs on-the-fly using the
    frozen Conv-AE encoder.  No quantum circuits are involved.
    """
    def __init__(self, image_loader, ae_encoder,
                 noise_std: float = NOISE_STD, max_batches: int = 100):
        super().__init__()
        self.noise_std    = noise_std
        self.clean_latents = []

        ae_encoder.eval()
        with torch.no_grad():
            for i, (imgs, _) in enumerate(image_loader):
                if i >= max_batches:
                    break
                z = ae_encoder(imgs.to(DEVICE)).cpu()   # (B, 64, 7, 7)
                self.clean_latents.append(z)

        self.clean_latents = torch.cat(self.clean_latents, dim=0)  # (N, 64, 7, 7)
        print(f"  Dataset: {len(self.clean_latents):,} latent maps  "
              f"(noise_std={noise_std})")

    def __len__(self):
        return len(self.clean_latents)

    def __getitem__(self, idx):
        clean = self.clean_latents[idx]
        noisy = clean + torch.randn_like(clean) * self.noise_std
        return noisy, clean


# ─────────────────────────── Training ────────────────────────────────────────
def train_stitching_cnn(epochs: int = EPOCHS_STITCH,
                         batch_size: int = BATCH_SIZE):
    """Train StitchingCNN on noisy→clean latent pairs (100% classical)."""

    # Load frozen Conv-AE encoder
    ae_model = ConvAutoencoder().to(DEVICE)
    if os.path.exists("models/conv_ae.pth"):
        ae_model.load_state_dict(torch.load("models/conv_ae.pth",
                                             map_location=DEVICE))
        print("  Loaded Conv-AE weights ✓")
    ae_model.eval()

    train_imgs, test_imgs = get_dataloaders(batch_size)

    print("\n[Building latent datasets (classical, no quantum) …]")
    train_ds = LatentRefinementDataset(train_imgs, ae_model.encoder,
                                        max_batches=100)
    test_ds  = LatentRefinementDataset(test_imgs,  ae_model.encoder,
                                        max_batches=20)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size,
                          shuffle=False, num_workers=0)

    model     = StitchingCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_STITCH)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  StitchingCNN parameters: {n_params:,}")
    print(f"\n[Training – {epochs} epochs …]")

    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        running = 0.0
        for noisy, clean in train_dl:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            pred = model(noisy)
            loss = criterion(pred, clean)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * noisy.size(0)
        train_losses.append(running / len(train_dl.dataset))

        # ── Eval ──
        model.eval()
        running = 0.0
        with torch.no_grad():
            for noisy, clean in test_dl:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                pred = model(noisy)
                loss = criterion(pred, clean)
                running += loss.item() * noisy.size(0)
        test_losses.append(running / len(test_dl.dataset))
        scheduler.step()

        if epoch % 3 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_losses[-1]:.6f} | "
                  f"Test : {test_losses[-1]:.6f}")

    return model, ae_model, train_losses, test_losses


# ─────────────────────────── Visualisation ───────────────────────────────────
def save_stitching_comparison(model, ae_model, test_loader, n: int = 10):
    """3-row grid: Original | Noisy-latent recon | Stitched (refined) recon."""
    model.eval(); ae_model.eval()
    imgs, _ = next(iter(test_loader))
    imgs = imgs[:n].to(DEVICE)

    with torch.no_grad():
        clean_z   = ae_model.encoder(imgs)
        noisy_z   = clean_z + torch.randn_like(clean_z) * NOISE_STD
        refined_z = model(noisy_z)

        orig_recon    = ae_model.decoder(clean_z)
        noisy_recon   = ae_model.decoder(noisy_z)
        refined_recon = ae_model.decoder(refined_z)

    rows   = [imgs, noisy_recon, refined_recon]
    titles = ["Original", f"Noisy Latent Recon (σ={NOISE_STD})",
              "Stitching CNN Refined Recon"]

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    for r, (row_imgs, title) in enumerate(zip(rows, titles)):
        rn = row_imgs.cpu().numpy()
        for c in range(n):
            axes[r, c].imshow(rn[c, 0] * 0.5 + 0.5, cmap="gray",
                              vmin=0, vmax=1, interpolation="none")
            axes[r, c].axis("off")
        axes[r, 0].set_title(title, fontsize=10, loc="left")

    plt.suptitle("Module 3 – Classical Stitching CNN", fontsize=12)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "stitching_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")
    return out


def save_latent_maps(model, ae_model, test_loader):
    """Visualise clean vs noisy vs refined latent (first 4 channels)."""
    model.eval(); ae_model.eval()
    imgs, _ = next(iter(test_loader))
    imgs = imgs[:1].to(DEVICE)

    with torch.no_grad():
        clean_z   = ae_model.encoder(imgs)
        noisy_z   = clean_z + torch.randn_like(clean_z) * NOISE_STD
        refined_z = model(noisy_z)

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for col in range(4):
        for row, (fmap, lbl) in enumerate(
                [(clean_z, "Clean"), (noisy_z, "Noisy"), (refined_z, "Refined")]):
            axes[row, col].imshow(fmap[0, col].cpu().numpy(),
                                   cmap="viridis")
            axes[row, col].set_title(f"{lbl} ch{col}", fontsize=8)
            axes[row, col].axis("off")

    plt.suptitle("Module 3 – Latent Feature Maps (ch 0-3)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "latent_maps.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def save_loss_curve(train_losses, test_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(test_losses,  label="Test MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Module 3 – Stitching CNN Training Curve")
    plt.legend(); plt.grid(True)
    out = os.path.join(SAVE_DIR, "loss_curve.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────── Entry Point ─────────────────────────────────────
def main():
    print("=" * 60)
    print("MODULE 3 – Classical Stitching CNN")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print("  (Purely classical – no quantum circuits in training)")

    model, ae_model, tr_loss, te_loss = train_stitching_cnn()

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")

    print("\n[Saving outputs]")
    _, test_loader = get_dataloaders(BATCH_SIZE)
    save_stitching_comparison(model, ae_model, test_loader)
    save_latent_maps(model, ae_model, test_loader)
    save_loss_curve(tr_loss, te_loss)

    final_mse = te_loss[-1]
    print(f"\n  Final Test MSE : {final_mse:.6f}")
    print("MODULE 3 COMPLETE ✓")
    return model, final_mse


if __name__ == "__main__":
    main()
