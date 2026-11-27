"""
Module 4: Classical Diffusion Baselines
=========================================
- Implements MLP-based latent patch denoiser
- Implements U-Net-based latent patch denoiser
- Trains both on noisy→clean latent patch pairs
- Reports MSE for each model as performance baseline
- Saves reconstructed images using each baseline
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module1_autoencoder import (ConvAutoencoder, get_dataloaders,
                                  DEVICE, LATENT_DIM, LATENT_CHANNELS)
from module2_quantum_patch_encoding import (extract_patches,
                                             PATCH_SIZE, STRIDE)

# ─────────────────────────── Configuration ───────────────────────────────────
NOISE_STDDEV  = 0.3          # Gaussian noise added to latent patches
EPOCHS        = 20
LR            = 1e-3
BATCH_SIZE    = 128
LATENT_HW     = 7
PATCH_DIM     = LATENT_CHANNELS * PATCH_SIZE ** 2   # 64*4=256
SAVE_DIR      = "outputs/module4"
os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs("models",  exist_ok=True)


# ─────────────────────────── Noisy Patch Dataset ─────────────────────────────
class LatentPatchDataset(Dataset):
    """
    Generates noisy patch → clean patch pairs on-the-fly using
    the Conv-AE encoder (kept frozen).
    """
    def __init__(self, image_loader, ae_encoder, noise_std=NOISE_STDDEV,
                 max_batches=50):
        super().__init__()
        self.noise_std  = noise_std
        self.clean_patches = []

        ae_encoder.eval()
        with torch.no_grad():
            for i, (imgs, _) in enumerate(image_loader):
                if i >= max_batches:
                    break
                z = ae_encoder(imgs.to(DEVICE)).cpu()   # (B, 64, 7, 7)
                patches, _, _ = extract_patches(z, PATCH_SIZE, STRIDE)
                # (B, NP, 256) → flatten to (B*NP, 256)
                self.clean_patches.append(
                    patches.view(-1, patches.shape[-1]))

        self.clean_patches = torch.cat(self.clean_patches, dim=0)  # (N, 256)

    def __len__(self):
        return len(self.clean_patches)

    def __getitem__(self, idx):
        clean = self.clean_patches[idx]
        noisy = clean + torch.randn_like(clean) * self.noise_std
        return noisy, clean


# ─────────────────────────── MLP Denoiser ────────────────────────────────────
class MLPDenoiser(nn.Module):
    """
    Simple 4-layer MLP for latent patch denoising.
    Input / Output : (B, PATCH_DIM)
    """
    def __init__(self, patch_dim: int = PATCH_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(patch_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, patch_dim),
        )
        # Skip connection
        self.skip = nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


# ─────────────────────────── U-Net Denoiser ──────────────────────────────────
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNetDenoiser(nn.Module):
    """
    Lightweight U-Net for spatial latent denoising.
    Input / Output : (B, LATENT_CHANNELS, 7, 7)
    """
    def __init__(self, channels: int = LATENT_CHANNELS):
        super().__init__()
        # Encoder path
        self.enc1 = UNetBlock(channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

        # Bottleneck (operates on 4×4)
        self.bottleneck = UNetBlock(128, 256)

        # Decoder path
        self.up1    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1   = UNetBlock(256, 128)   # 128 skip + 128 up
        self.up2    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2   = UNetBlock(128, 64)    # 64 skip + 64 up

        self.out_conv = nn.Conv2d(64, channels, kernel_size=1)

    def forward(self, x):
        # x : (B, 64, 7, 7)
        e1 = self.enc1(x)                    # (B, 64, 7, 7)
        e2 = self.enc2(self.pool(e1))        # (B, 128, 4, 4)
        b  = self.bottleneck(self.pool(e2))  # (B, 256, 2, 2)

        d1 = self.up1(b)                     # (B, 128, 4, 4)
        # Align spatial dims if ceil_mode causes mismatch
        d1 = self._pad_to(d1, e2)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))

        d2 = self.up2(d1)                    # (B, 64, 8, 8)
        d2 = self._pad_to_crop(d2, e1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return self.out_conv(d2)             # (B, 64, 7, 7)

    @staticmethod
    def _pad_to(x, ref):
        """Pad x to match ref's spatial size."""
        dh = ref.shape[2] - x.shape[2]
        dw = ref.shape[3] - x.shape[3]
        if dh > 0 or dw > 0:
            x = nn.functional.pad(x, [0, dw, 0, dh])
        return x

    @staticmethod
    def _pad_to_crop(x, ref):
        """Crop or pad x to match ref's spatial size."""
        th, tw = ref.shape[2], ref.shape[3]
        x = x[:, :, :th, :tw]
        dh = th - x.shape[2]; dw = tw - x.shape[3]
        if dh > 0 or dw > 0:
            x = nn.functional.pad(x, [0, dw, 0, dh])
        return x


import torch.nn.functional as functional   # alias
UNetDenoiser._pad_to       = staticmethod(lambda x, ref: (
    functional.pad(x, [0, ref.shape[3]-x.shape[3], 0, ref.shape[2]-x.shape[2]])
    if (ref.shape[2]-x.shape[2] > 0 or ref.shape[3]-x.shape[3] > 0) else x
))
UNetDenoiser._pad_to_crop  = staticmethod(lambda x, ref: (
    functional.pad(x[:, :, :ref.shape[2], :ref.shape[3]],
                   [0, max(0, ref.shape[3]-x.shape[3]),
                    0, max(0, ref.shape[2]-x.shape[2])])
))


# ─────────────────────────── Training Utility ────────────────────────────────
def train_model(model, train_loader, test_loader, model_name="model",
                epochs=EPOCHS, lr=LR, is_spatial=False):
    """
    Generic training loop for both MLP and U-Net denoisers.
    is_spatial : True → U-Net (input is (B,64,7,7)); False → MLP (input is (B,256))
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    model.to(DEVICE)

    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        running = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            if is_spatial:
                # Reshape patches back to spatial maps for U-Net
                B = noisy.shape[0]
                # We're feeding single-patch spatial maps padded to 7×7 when needed
                # but the dataset returns flat patches – handle both shapes
                if noisy.ndim == 2:
                    noisy = noisy.view(B, LATENT_CHANNELS, PATCH_SIZE, PATCH_SIZE)
                    clean = clean.view(B, LATENT_CHANNELS, PATCH_SIZE, PATCH_SIZE)
                    # Pad to 7×7 for U-Net spatial ops
                    noisy = functional.pad(noisy, [0, LATENT_HW - PATCH_SIZE,
                                                   0, LATENT_HW - PATCH_SIZE])
                    clean = functional.pad(clean, [0, LATENT_HW - PATCH_SIZE,
                                                   0, LATENT_HW - PATCH_SIZE])

            pred = model(noisy)
            loss = criterion(pred, clean)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * noisy.size(0)
        train_losses.append(running / len(train_loader.dataset))

        # ── Eval ──
        model.eval()
        running = 0.0
        with torch.no_grad():
            for noisy, clean in test_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                if is_spatial and noisy.ndim == 2:
                    B = noisy.shape[0]
                    noisy = functional.pad(
                        noisy.view(B, LATENT_CHANNELS, PATCH_SIZE, PATCH_SIZE),
                        [0, LATENT_HW - PATCH_SIZE, 0, LATENT_HW - PATCH_SIZE])
                    clean = functional.pad(
                        clean.view(B, LATENT_CHANNELS, PATCH_SIZE, PATCH_SIZE),
                        [0, LATENT_HW - PATCH_SIZE, 0, LATENT_HW - PATCH_SIZE])
                pred = model(noisy)
                loss = criterion(pred, clean)
                running += loss.item() * noisy.size(0)
        test_losses.append(running / len(test_loader.dataset))

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{model_name}] Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_losses[-1]:.6f} | Test: {test_losses[-1]:.6f}")

    return train_losses, test_losses


# ─────────────────────────── U-Net Spatial Training ──────────────────────────
class FullLatentDataset(Dataset):
    """Dataset of full (noisy, clean) latent maps (B, 64, 7, 7) for U-Net."""
    def __init__(self, image_loader, ae_encoder, noise_std=NOISE_STDDEV,
                 max_batches=50):
        super().__init__()
        self.noise_std = noise_std
        self.clean_latents = []

        ae_encoder.eval()
        with torch.no_grad():
            for i, (imgs, _) in enumerate(image_loader):
                if i >= max_batches:
                    break
                z = ae_encoder(imgs.to(DEVICE)).cpu()
                self.clean_latents.append(z)
        self.clean_latents = torch.cat(self.clean_latents, dim=0)

    def __len__(self):
        return len(self.clean_latents)

    def __getitem__(self, idx):
        clean = self.clean_latents[idx]
        noisy = clean + torch.randn_like(clean) * self.noise_std
        return noisy, clean


def train_unet(ae_model, train_loader_imgs, test_loader_imgs,
               epochs=EPOCHS, lr=LR):
    """Train U-Net on full latent maps (not patches)."""
    train_ds = FullLatentDataset(train_loader_imgs, ae_model.encoder)
    test_ds  = FullLatentDataset(test_loader_imgs,  ae_model.encoder, max_batches=10)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    unet = UNetDenoiser().to(DEVICE)
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        unet.train()
        running = 0.0
        for noisy, clean in train_dl:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            pred = unet(noisy)
            loss = criterion(pred, clean)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * noisy.size(0)
        train_losses.append(running / len(train_dl.dataset))

        unet.eval()
        running = 0.0
        with torch.no_grad():
            for noisy, clean in test_dl:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                pred = unet(noisy)
                loss = criterion(pred, clean)
                running += loss.item() * noisy.size(0)
        test_losses.append(running / len(test_dl.dataset))
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [U-Net] Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_losses[-1]:.6f} | Test: {test_losses[-1]:.6f}")

    return unet, train_losses, test_losses


# ─────────────────────────── Reconstruction Visuals ──────────────────────────
def save_baseline_reconstructions(mlp, unet, ae_model, test_loader, n=10):
    """
    Show: Original | Noisy AE recon | MLP-denoised recon | U-Net-denoised recon
    """
    ae_model.eval(); unet.eval(); mlp.eval()
    imgs, _ = next(iter(test_loader))
    imgs = imgs[:n].to(DEVICE)
    noise_std = NOISE_STDDEV

    with torch.no_grad():
        clean_z = ae_model.encoder(imgs)                     # (N, 64, 7, 7)
        noisy_z = clean_z + torch.randn_like(clean_z) * noise_std

        # MLP: operate on flattened patches then reconstruct
        p, n_h, n_w = extract_patches(noisy_z.cpu(), PATCH_SIZE, STRIDE)
        p = p.to(DEVICE)
        mlp_p = mlp(p.view(-1, p.shape[-1])).view(p.shape)   # (N, NP, 256)

        # Fold patches back to latent space preserving (C, P, P) spatial blocks
        P = PATCH_SIZE
        C = LATENT_CHANNELS
        H = (n_h - 1) * STRIDE + P
        W = (n_w - 1) * STRIDE + P
        mlp_sp = mlp_p.view(n, mlp_p.shape[1], C, P, P).cpu()
        mlp_z  = torch.zeros(n, C, H, W)
        count  = torch.zeros(n, 1, H, W)
        idx = 0
        for r in range(n_h):
            for c in range(n_w):
                rs, cs = r * STRIDE, c * STRIDE
                mlp_z[:, :, rs:rs+P, cs:cs+P] += mlp_sp[:, idx]
                count[:, :, rs:rs+P, cs:cs+P] += 1.0
                idx += 1
        mlp_z = (mlp_z / count.clamp(min=1)).to(DEVICE)

        unet_z     = unet(noisy_z)

        noisy_recon = ae_model.decoder(noisy_z)
        mlp_recon   = ae_model.decoder(mlp_z)
        unet_recon  = ae_model.decoder(unet_z)
        clean_recon = ae_model.decoder(clean_z)

    rows = [imgs, noisy_recon, mlp_recon, unet_recon, clean_recon]
    titles = ["Original", "Noisy Latent", "MLP Denoised", "U-Net Denoised", "Clean Latent"]

    fig, axes = plt.subplots(5, n, figsize=(2 * n, 10))
    for r, (row_imgs, title) in enumerate(zip(rows, titles)):
        row_np = row_imgs.cpu().numpy()
        for c in range(n):
            axes[r, c].imshow(row_np[c, 0] * 0.5 + 0.5, cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_title(title, fontsize=10, loc="left")

    plt.suptitle("Module 4 – Classical Diffusion Baselines", fontsize=13)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "baseline_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")
    return out


def save_mse_bar_chart(mse_dict: dict):
    fig, ax = plt.subplots(figsize=(7, 4))
    names  = list(mse_dict.keys())
    values = list(mse_dict.values())
    bars = ax.bar(names, values, color=["steelblue", "darkorange", "forestgreen"])
    ax.bar_label(bars, fmt="%.5f", padding=3)
    ax.set_ylabel("Test MSE"); ax.set_title("Module 4 – Baseline MSE Comparison")
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(SAVE_DIR, "mse_comparison.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────── Entry Point ─────────────────────────────────────
def main():
    print("=" * 60)
    print("MODULE 4 – Classical Diffusion Baselines (MLP + U-Net)")
    print("=" * 60)

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # Load pre-trained Conv-AE
    ae_model = ConvAutoencoder().to(DEVICE)
    if os.path.exists("models/conv_ae.pth"):
        ae_model.load_state_dict(torch.load("models/conv_ae.pth",
                                             map_location=DEVICE))
        print("  Loaded Conv-AE ✓")
    ae_model.eval()

    # ── Build patch datasets ──
    print("\n[Building patch datasets …]")
    train_patch_ds = LatentPatchDataset(train_loader, ae_model.encoder, max_batches=50)
    test_patch_ds  = LatentPatchDataset(test_loader,  ae_model.encoder, max_batches=10)
    train_patch_dl = DataLoader(train_patch_ds, batch_size=BATCH_SIZE,
                                shuffle=True,  num_workers=0)
    test_patch_dl  = DataLoader(test_patch_ds,  batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=0)
    print(f"  Train patches : {len(train_patch_ds):,}  |  Test patches : {len(test_patch_ds):,}")

    # ── MLP Denoiser ──
    print("\n[Training MLP Denoiser]")
    mlp = MLPDenoiser(PATCH_DIM).to(DEVICE)
    mlp_tr, mlp_te = train_model(mlp, train_patch_dl, test_patch_dl,
                                  model_name="MLP", epochs=EPOCHS)
    torch.save(mlp.state_dict(), "models/mlp_denoiser.pth")
    print("  MLP saved → models/mlp_denoiser.pth")

    # ── U-Net Denoiser ──
    print("\n[Training U-Net Denoiser]")
    unet, unet_tr, unet_te = train_unet(ae_model, train_loader, test_loader,
                                         epochs=EPOCHS)
    torch.save(unet.state_dict(), "models/unet_denoiser.pth")
    print("  U-Net saved → models/unet_denoiser.pth")

    # ── MSE Summary ──
    mse_mlp  = mlp_te[-1]
    mse_unet = unet_te[-1]
    print(f"\n  MLP   Test MSE : {mse_mlp:.6f}")
    print(f"  U-Net Test MSE : {mse_unet:.6f}")
    mse_dict = {"MLP": mse_mlp, "U-Net": mse_unet}

    # ── Visuals ──
    print("\n[Saving outputs]")
    save_baseline_reconstructions(mlp, unet, ae_model, test_loader)
    save_mse_bar_chart(mse_dict)

    # Combined loss curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(mlp_tr, label="Train"); ax1.plot(mlp_te, label="Test")
    ax1.set_title("MLP Denoiser"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE")
    ax1.legend(); ax1.grid(True)
    ax2.plot(unet_tr, label="Train"); ax2.plot(unet_te, label="Test")
    ax2.set_title("U-Net Denoiser"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True)
    plt.suptitle("Module 4 – Classical Baseline Training Curves", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print("\nMODULE 4 COMPLETE ✓")
    return mlp, unet, mse_dict


if __name__ == "__main__":
    main()
