"""
PHQD Quick Test Script
========================
Runs a minimal smoke-test of all 5 modules with tiny epochs
to verify everything imports and runs without error.

Usage:
    python test_phqd.py
"""

import os, sys, torch
import numpy as np

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

DEVICE = torch.device("cpu")
print(f"Running PHQD smoke-test on: {DEVICE}\n")

# ─── Module 1 ──────────────────────────────────────────────────────────────
print("=" * 50)
print("TEST: Module 1 – Conv-AE")
from module1_autoencoder import (ConvAutoencoder, get_dataloaders,
                                  LATENT_DIM, DEVICE as DEV1)
model_ae = ConvAutoencoder().to(DEVICE)
train_l, test_l = get_dataloaders(batch_size=32)
imgs, _ = next(iter(test_l))
imgs = imgs[:4].to(DEVICE)
recon, z = model_ae(imgs)
assert recon.shape == imgs.shape,  f"Recon shape mismatch: {recon.shape}"
assert z.shape == (4, 64, 7, 7),   f"Latent shape mismatch: {z.shape}"
torch.save(model_ae.state_dict(), "models/conv_ae.pth")
print(f"  ✓ Encoder output: {z.shape}")
print(f"  ✓ Decoder output: {recon.shape}")

# ─── Module 2 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST: Module 2 – Quantum Patch Encoding")
from module2_quantum_patch_encoding import (extract_patches, QuantumPatchEncoder,
                                             PATCH_SIZE, STRIDE, N_QUBITS,
                                             LATENT_CHANNELS)
patches, n_h, n_w = extract_patches(z.cpu(), PATCH_SIZE, STRIDE)
print(f"  ✓ Patches shape: {patches.shape}  ({n_h}×{n_w} grid)")

qpe = QuantumPatchEncoder(in_features=patches.shape[2])
# Only 1 patch to keep smoke test fast
with torch.no_grad():
    q_out = qpe(patches[:1, :1, :])   # (1, 1, N_QUBITS)
assert q_out.shape == (1, 1, N_QUBITS), f"QPE output shape: {q_out.shape}"
print(f"  ✓ Quantum encoded patch: {q_out.shape}")

# ─── Module 3 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST: Module 3 – Stitching CNN")
from module3_stitching_cnn import (patches_to_latent, StitchingCNN, LATENT_HW)
# Test fold-back using dummy patch features (N_QUBITS channels → used for stitch utility)
dummy_q  = torch.randn(2, n_h * n_w, N_QUBITS)
stitched = patches_to_latent(dummy_q, n_h, n_w, PATCH_SIZE, STRIDE, N_QUBITS)
print(f"  ✓ Stitched map: {stitched.shape}")

# Test StitchingCNN with full latent-channel input (normal use-case)
dummy_latent_input = torch.randn(2, LATENT_CHANNELS, LATENT_HW, LATENT_HW)
cnn     = StitchingCNN(in_channels=LATENT_CHANNELS, out_channels=LATENT_CHANNELS)
refined = cnn(dummy_latent_input)
assert refined.shape == (2, LATENT_CHANNELS, LATENT_HW, LATENT_HW), \
    f"Stitch CNN output: {refined.shape}"
print(f"  ✓ Refined latent: {refined.shape}")

# ─── Module 4 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST: Module 4 – Classical Baselines")
from module4_classical_baselines import (MLPDenoiser, UNetDenoiser, PATCH_DIM)
mlp = MLPDenoiser(PATCH_DIM)
dummy_patch = torch.randn(4, PATCH_DIM)
mlp_out = mlp(dummy_patch)
assert mlp_out.shape == (4, PATCH_DIM), f"MLP output: {mlp_out.shape}"
print(f"  ✓ MLP output: {mlp_out.shape}")

unet = UNetDenoiser()
dummy_latent = torch.randn(2, LATENT_CHANNELS, 7, 7)
unet_out = unet(dummy_latent)
assert unet_out.shape == dummy_latent.shape, f"U-Net output: {unet_out.shape}"
print(f"  ✓ U-Net output: {unet_out.shape}")

# ─── Module 5 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST: Module 5 – EPQD")
from module5_epqd import EPQDDenoiser
epqd = EPQDDenoiser(name="test", n_qubits=4, n_layers=1, ansatz="basic",
                     patch_dim=PATCH_DIM)
dummy_patch = torch.randn(2, PATCH_DIM)
epqd_out = epqd(dummy_patch)
assert epqd_out.shape == (2, PATCH_DIM), f"EPQD output: {epqd_out.shape}"
print(f"  ✓ EPQD output: {epqd_out.shape}")

# ─── All Passed ────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("✅  All module smoke-tests PASSED!")
print("=" * 50)
print("\nRun the full pipeline with:")
print("   python run_phqd.py")
print("Or test a single module:")
print("   python run_phqd.py --module 1")
