"""
Module 1: Classical Autoencoder (Conv-AE)
==========================================
- Loads and preprocesses Fashion-MNIST images
- Compresses images into latent representations via encoder
- Reconstructs images using the decoder
- Saves model weights and reconstructed image samples
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────── Configuration ───────────────────────────────────
LATENT_DIM      = 64          # Latent space channels after encoding
LATENT_CHANNELS = 64          # Alias used by downstream modules
EPOCHS       = 20
BATCH_SIZE   = 128
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR     = "outputs/module1"
MODEL_PATH   = "models/conv_ae.pth"
os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ─────────────────────────── Dataset ─────────────────────────────────────────
def get_dataloaders(batch_size=BATCH_SIZE):
    """Return train and test DataLoaders for Fashion-MNIST (normalised to [-1,1])."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = torchvision.datasets.FashionMNIST(
        root="data", train=True,  download=True, transform=tf)
    test_ds  = torchvision.datasets.FashionMNIST(
        root="data", train=False, download=True, transform=tf)
    _pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=_pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=_pin)
    return train_loader, test_loader

# ─────────────────────────── Architecture ────────────────────────────────────
class Encoder(nn.Module):
    """
    Conv encoder: 1×28×28  →  64×7×7 latent feature map
    Two strided conv blocks with BatchNorm + LeakyReLU.
    """
    def __init__(self, latent_channels=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            # 1×28×28 → 32×14×14
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 32×14×14 → 64×7×7
            nn.Conv2d(32, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)       # (B, 64, 7, 7)


class Decoder(nn.Module):
    """
    Conv decoder: 64×7×7  →  1×28×28 reconstructed image
    Two transposed-conv blocks.
    """
    def __init__(self, latent_channels=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            # 64×7×7 → 32×14×14
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32×14×14 → 1×28×28
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)       # (B, 1, 28, 28)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_channels=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# ─────────────────────────── Training ────────────────────────────────────────
def train_autoencoder(model, train_loader, test_loader, epochs=EPOCHS):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)
            recon, _ = model(imgs)
            loss = criterion(recon, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # ── Eval ──
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(DEVICE)
                recon, _ = model(imgs)
                loss = criterion(recon, imgs)
                running_loss += loss.item() * imgs.size(0)
        test_loss = running_loss / len(test_loader.dataset)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Train MSE: {train_loss:.6f} | "
                  f"Test MSE: {test_loss:.6f}")

    return train_losses, test_losses

# ─────────────────────────── Visualisation ───────────────────────────────────
def save_reconstruction_grid(model, test_loader, n=10, tag=""):
    """Save a grid of original vs reconstructed images (high quality, 300 dpi)."""
    model.eval()
    imgs, _ = next(iter(test_loader))
    imgs = imgs[:n].to(DEVICE)
    with torch.no_grad():
        recons, _ = model(imgs)

    imgs   = imgs.cpu().numpy()
    recons = recons.cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(imgs[i, 0] * 0.5 + 0.5, cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i, 0] * 0.5 + 0.5, cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", fontsize=10, loc="left")
    axes[1, 0].set_title("Reconstructed", fontsize=10, loc="left")
    fig.suptitle(f"Module 1 – Conv-AE Reconstruction{tag}", fontsize=12)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, f"reconstruction{tag}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")
    return out


def save_loss_curve(train_losses, test_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(test_losses,  label="Test MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Module 1 – Conv-AE Training Curve")
    plt.legend(); plt.grid(True)
    out = os.path.join(SAVE_DIR, "loss_curve.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def extract_and_save_latents(model, loader, max_batches=5, split="train"):
    """Extract latent vectors and save as .npy for downstream modules."""
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            if i >= max_batches:
                break
            z = model.encoder(imgs.to(DEVICE))
            latents.append(z.cpu().numpy())
            labels.append(lbls.numpy())
    latents = np.concatenate(latents, axis=0)   # (N, 64, 7, 7)
    labels  = np.concatenate(labels,  axis=0)
    np.save(f"outputs/latents_{split}.npy", latents)
    np.save(f"outputs/labels_{split}.npy",  labels)
    print(f"  Latent shape ({split}): {latents.shape}")
    return latents, labels


# ─────────────────────────── Entry Point ─────────────────────────────────────
def main():
    print("=" * 60)
    print("MODULE 1 – Classical Convolutional Autoencoder (Conv-AE)")
    print("=" * 60)
    print(f"  Device : {DEVICE}")

    train_loader, test_loader = get_dataloaders()

    model = ConvAutoencoder().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}")

    # ── Training ──
    print("\n[Training]")
    train_losses, test_losses = train_autoencoder(model, train_loader, test_loader)

    # ── Save model ──
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")

    # ── Outputs ──
    print("\n[Outputs]")
    save_reconstruction_grid(model, test_loader)
    save_loss_curve(train_losses, test_losses)

    os.makedirs("outputs", exist_ok=True)
    extract_and_save_latents(model, train_loader, max_batches=20, split="train")
    extract_and_save_latents(model, test_loader,  max_batches=5,  split="test")

    final_mse = test_losses[-1]
    print(f"\n  Final Test MSE : {final_mse:.6f}")
    print("MODULE 1 COMPLETE ✓")
    return final_mse


if __name__ == "__main__":
    main()
