"""
PHQD FID Evaluation Script
===========================
Computes standard InceptionV3 Fréchet Inception Distance (FID) on 1,000
Fashion-MNIST test images across Conv-AE, Noisy, MLP, U-Net, and all 4 EPQD
quantum configurations.
"""

import os
import json
import time
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

from module1_autoencoder import ConvAutoencoder, get_dataloaders, LATENT_CHANNELS
from module2_quantum_patch_encoding import extract_patches, PATCH_SIZE, STRIDE
from module4_classical_baselines import MLPDenoiser, UNetDenoiser, NOISE_STDDEV, PATCH_DIM
from module5_epqd import EPQDDenoiser, fold_patches_spatial, QC_CONFIGS

DEVICE = torch.device("cpu")
N_SAMPLES = 1000
BATCH_SIZE = 64
torch.manual_seed(42)
np.random.seed(42)


def get_inception_feature_extractor():
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = nn.Identity()
    model.eval()
    return model.to(DEVICE)


@torch.no_grad()
def extract_inception_features(imgs_neg1_to_1: torch.Tensor, inc_model: nn.Module, batch_size: int = 64) -> np.ndarray:
    """
    imgs_neg1_to_1: (N, 1, 28, 28) tensor in [-1, 1].
    Converts to [0, 1], 3-channel RGB, resizes to 299x299, normalizes with ImageNet stats,
    and extracts 2048-D InceptionV3 pool3 features.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    feats = []
    for i in range(0, len(imgs_neg1_to_1), batch_size):
        batch = (imgs_neg1_to_1[i : i + batch_size] * 0.5 + 0.5).clamp(0.0, 1.0)
        batch = batch.repeat(1, 3, 1, 1)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch - mean) / std
        f = inc_model(batch)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def calculate_fid(feat_real: np.ndarray, feat_gen: np.ndarray) -> float:
    mu1, sigma1 = np.mean(feat_real, axis=0), np.cov(feat_real, rowvar=False)
    mu2, sigma2 = np.mean(feat_gen, axis=0), np.cov(feat_gen, rowvar=False)

    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


def main():
    t0 = time.time()
    print("=" * 65)
    print(f"PHQD FID EVALUATION (N={N_SAMPLES} Fashion-MNIST Test Images)")
    print("=" * 65)

    _, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    imgs_list = []
    for x, _ in test_loader:
        imgs_list.append(x)
        if sum(b.shape[0] for b in imgs_list) >= N_SAMPLES:
            break
    real_imgs = torch.cat(imgs_list, dim=0)[:N_SAMPLES].to(DEVICE)
    print(f"Loaded {real_imgs.shape[0]} test images.")

    # Load models
    ae = ConvAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load("models/conv_ae.pth", map_location=DEVICE))
    ae.eval()

    mlp = MLPDenoiser(PATCH_DIM).to(DEVICE)
    mlp.load_state_dict(torch.load("models/mlp_denoiser.pth", map_location=DEVICE))
    mlp.eval()

    unet = UNetDenoiser().to(DEVICE)
    unet.load_state_dict(torch.load("models/unet_denoiser.pth", map_location=DEVICE))
    unet.eval()

    epqd_models = {}
    for cfg in QC_CONFIGS:
        m = EPQDDenoiser(
            name=cfg["name"],
            n_qubits=cfg["n_qubits"],
            n_layers=cfg["n_layers"],
            ansatz=cfg["ansatz"],
            patch_dim=PATCH_DIM,
        ).to(DEVICE)
        m.load_state_dict(torch.load(f"models/epqd_{cfg['name']}.pth", map_location=DEVICE))
        m.eval()
        epqd_models[cfg["name"]] = m

    print("Loading InceptionV3 (ImageNet weights)...")
    inc = get_inception_feature_extractor()
    feat_real = extract_inception_features(real_imgs, inc, BATCH_SIZE)
    print(f"Real features extracted: {feat_real.shape} ({time.time() - t0:.1f}s)")

    # Generate latents and reconstructions in batches
    clean_imgs, noisy_imgs, mlp_imgs, unet_imgs = [], [], [], []
    epqd_imgs = {k: [] for k in epqd_models}

    with torch.no_grad():
        for i in range(0, N_SAMPLES, BATCH_SIZE):
            b_imgs = real_imgs[i : i + BATCH_SIZE]
            B = b_imgs.shape[0]
            clean_z = ae.encoder(b_imgs)
            noisy_z = clean_z + torch.randn_like(clean_z) * NOISE_STDDEV

            clean_imgs.append(ae.decoder(clean_z))
            noisy_imgs.append(ae.decoder(noisy_z))

            p, nh, nw = extract_patches(noisy_z, PATCH_SIZE, STRIDE)
            NP, D = p.shape[1], p.shape[2]

            # MLP
            mp = mlp(p.view(B * NP, D)).view(B, NP, D)
            mz = fold_patches_spatial(mp, nh, nw, PATCH_SIZE, STRIDE, LATENT_CHANNELS)
            mlp_imgs.append(ae.decoder(mz))

            # U-Net
            unet_imgs.append(ae.decoder(unet(noisy_z)))

            # EPQD configs
            for cname, em in epqd_models.items():
                ep = em(p.view(B * NP, D)).view(B, NP, D)
                ez = fold_patches_spatial(ep, nh, nw, PATCH_SIZE, STRIDE, LATENT_CHANNELS)
                epqd_imgs[cname].append(ae.decoder(ez))

    candidates = {
        "Conv-AE Clean": torch.cat(clean_imgs, dim=0),
        "Noisy (sigma=0.3)": torch.cat(noisy_imgs, dim=0),
        "MLP": torch.cat(mlp_imgs, dim=0),
        "U-Net": torch.cat(unet_imgs, dim=0),
    }
    for cname, lst in epqd_imgs.items():
        candidates[cname] = torch.cat(lst, dim=0)

    fid_results = {}
    print("\n" + "-" * 45)
    print(f"  {'Model':<22} {'FID (Lower=Better)':>18}")
    print("-" * 45)
    for name, gen_imgs in candidates.items():
        feat_gen = extract_inception_features(gen_imgs, inc, BATCH_SIZE)
        fid = calculate_fid(feat_real, feat_gen)
        fid_results[name] = fid
        print(f"  {name:<22} {fid:18.4f}")
    print("-" * 45)
    print(f"Total elapsed: {time.time() - t0:.1f}s")

    os.makedirs("outputs/summary", exist_ok=True)
    with open("outputs/summary/fid_results.json", "w") as f:
        json.dump({"n_samples": N_SAMPLES, "fid": fid_results}, f, indent=2)
    print("Saved -> outputs/summary/fid_results.json")


if __name__ == "__main__":
    main()
