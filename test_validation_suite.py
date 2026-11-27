"""
PHQD Complete Validation Suite: Unit, Integration, and System Testing
====================================================================
Executes real tests against the existing PHQD modules and trained checkpoints
without modifying saved model weights or experimental outputs.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from module1_autoencoder import ConvAutoencoder, get_dataloaders, LATENT_CHANNELS
from module2_quantum_patch_encoding import (
    extract_patches,
    patches_to_angles,
    QuantumPatchEncoder,
    PATCH_SIZE,
    STRIDE,
    N_QUBITS,
)
from module3_stitching_cnn import patches_to_latent, StitchingCNN, LATENT_HW
from module4_classical_baselines import (
    MLPDenoiser,
    UNetDenoiser,
    NOISE_STDDEV,
    PATCH_DIM,
)
from module5_epqd import (
    EPQDDenoiser,
    fold_patches_spatial,
    epqd_full_reconstruction,
    QC_CONFIGS,
)

DEVICE = torch.device("cpu")
torch.manual_seed(42)
np.random.seed(42)


# ─────────────────────────── Image Quality Metrics ───────────────────────────
def to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] image tensor to [0, 1] clamped."""
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.mse_loss(pred, target).item())


def compute_psnr(pred_01: torch.Tensor, target_01: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(pred_01, target_01).item()
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_ssim(pred_01: torch.Tensor, target_01: torch.Tensor, window_size: int = 7) -> float:
    """Standard SSIM over (B, 1, H, W) tensors in [0, 1]."""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    # Uniform/Gaussian-like averaging window
    coords = torch.arange(window_size, dtype=torch.float32, device=pred_01.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    pad = window_size // 2
    mu1 = F.conv2d(pred_01, window, padding=pad)
    mu2 = F.conv2d(target_01, window, padding=pad)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_01 * pred_01, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target_01 * target_01, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(pred_01 * target_01, window, padding=pad) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean().item())


# ─────────────────────────── Load Trained Models ─────────────────────────────
def load_models():
    ae = ConvAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load("models/conv_ae.pth", map_location=DEVICE))
    ae.eval()

    qpe = QuantumPatchEncoder(in_features=PATCH_DIM, n_qubits=N_QUBITS, n_layers=2).to(DEVICE)
    if os.path.exists("models/quantum_encoder.pt"):
        qpe.load_state_dict(torch.load("models/quantum_encoder.pt", map_location=DEVICE))
    qpe.eval()

    stitch = StitchingCNN().to(DEVICE)
    stitch.load_state_dict(torch.load("models/stitching_cnn.pth", map_location=DEVICE))
    stitch.eval()

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

    return ae, qpe, stitch, mlp, unet, epqd_models


def main():
    t_start = time.time()
    print("=" * 75)
    print("PHQD COMPLETE VALIDATION SUITE (UNIT, INTEGRATION, SYSTEM TESTING)")
    print("=" * 75)

    _, test_loader = get_dataloaders(batch_size=32)
    test_imgs, test_labels = next(iter(test_loader))  # 32 real Fashion-MNIST test images
    test_imgs = test_imgs.to(DEVICE)

    ae, qpe, stitch, mlp, unet, epqd_models = load_models()
    print(f"Loaded Fashion-MNIST test batch: {tuple(test_imgs.shape)} | Models loaded: 9")

    results = {"unit": [], "integration": [], "system": []}

    # ══════════════════════════════════════════════════════════════════════════
    # 1. UNIT TESTING (UT01 - UT09)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 75)
    print("1. UNIT TESTING")
    print("─" * 75)

    # UT01: Data Pipeline & Image Preprocessing
    n_samples = 32
    imgs_u1 = test_imgs[:n_samples]
    assert imgs_u1.shape == (32, 1, 28, 28)
    min_v, max_v = float(imgs_u1.min()), float(imgs_u1.max())
    assert min_v >= -1.0001 and max_v <= 1.0001
    ut01_res = f"Shape {tuple(imgs_u1.shape)}, range [{min_v:.3f}, {max_v:.3f}], dtype={imgs_u1.dtype}"
    print(f"[UT01] Pass | {ut01_res}")
    results["unit"].append(("UT01", "Pass", ut01_res))

    # UT02: Latent Encoding & Decoding (Conv-AE)
    with torch.no_grad():
        recons_u2, z_u2 = ae(imgs_u1)
    assert z_u2.shape == (32, 64, 7, 7)
    assert recons_u2.shape == (32, 1, 28, 28)
    ae_mse = compute_mse(recons_u2, imgs_u1)
    ut02_res = f"Latent {tuple(z_u2.shape)}, Recon {tuple(recons_u2.shape)}, AE MSE={ae_mse:.6f}"
    print(f"[UT02] Pass | {ut02_res}")
    results["unit"].append(("UT02", "Pass", ut02_res))

    # UT03: Overlapping Patch Extraction
    z_sub = z_u2[:16]
    patches_u3, n_h, n_w = extract_patches(z_sub, PATCH_SIZE, STRIDE)
    assert patches_u3.shape == (16, 36, 256) and (n_h, n_w) == (6, 6)
    ut03_res = f"Patches {tuple(patches_u3.shape)}, grid={n_h}x{n_w} (36 overlapping 2x2 patches, stride=1)"
    print(f"[UT03] Pass | {ut03_res}")
    results["unit"].append(("UT03", "Pass", ut03_res))

    # UT04: Quantum Encoding (PQC)
    angles_u4 = patches_to_angles(patches_u3[:4])
    assert float(angles_u4.min()) >= 0.0 and float(angles_u4.max()) <= np.pi + 1e-5
    with torch.no_grad():
        q_enc_u4 = qpe(patches_u3[:4])  # (4, 36, 4)
    assert q_enc_u4.shape == (4, 36, N_QUBITS)
    q_min, q_max = float(q_enc_u4.min()), float(q_enc_u4.max())
    assert q_min >= -1.0001 and q_max <= 1.0001
    ut04_res = f"4-qubit PQC output {tuple(q_enc_u4.shape)}, expectation range [{q_min:.4f}, {q_max:.4f}] (Note: 4 qubits used instead of 6 in PPT)"
    print(f"[UT04] Pass | {ut04_res}")
    results["unit"].append(("UT04", "Pass", ut04_res))

    # UT05: Classical Stitching CNN & Patch Fold-Back
    noisy_z_u5 = z_sub + torch.randn_like(z_sub) * 0.15
    with torch.no_grad():
        refined_u5 = stitch(noisy_z_u5)
    assert refined_u5.shape == (16, 64, 7, 7)
    mse_before_u5 = compute_mse(noisy_z_u5, z_sub)
    mse_after_u5 = compute_mse(refined_u5, z_sub)
    assert mse_after_u5 < mse_before_u5
    ut05_res = f"Refined {tuple(refined_u5.shape)}, Latent MSE reduced {mse_before_u5:.6f} -> {mse_after_u5:.6f}"
    print(f"[UT05] Pass | {ut05_res}")
    results["unit"].append(("UT05", "Pass", ut05_res))

    # UT06: Classical Diffusion Baselines (MLP & U-Net)
    noisy_z_u6 = z_sub + torch.randn_like(z_sub) * NOISE_STDDEV
    noisy_p_u6, _, _ = extract_patches(noisy_z_u6, PATCH_SIZE, STRIDE)
    clean_p_u6, _, _ = extract_patches(z_sub, PATCH_SIZE, STRIDE)
    with torch.no_grad():
        mlp_p_u6 = mlp(noisy_p_u6.view(-1, PATCH_DIM)).view_as(noisy_p_u6)
        unet_z_u6 = unet(noisy_z_u6)
    noisy_p_mse = compute_mse(noisy_p_u6, clean_p_u6)
    mlp_p_mse = compute_mse(mlp_p_u6, clean_p_u6)
    noisy_z_mse = compute_mse(noisy_z_u6, z_sub)
    unet_z_mse = compute_mse(unet_z_u6, z_sub)
    assert mlp_p_mse < noisy_p_mse and unet_z_mse < noisy_z_mse
    ut06_res = f"MLP patch MSE {noisy_p_mse:.5f}->{mlp_p_mse:.5f}; U-Net latent MSE {noisy_z_mse:.5f}->{unet_z_mse:.5f}"
    print(f"[UT06] Pass | {ut06_res}")
    results["unit"].append(("UT06", "Pass", ut06_res))

    # UT07: EPQD Diffusion Process (Quantum Patch Denoising)
    epqd_mses = {}
    with torch.no_grad():
        for name, em in epqd_models.items():
            den_p = em(noisy_p_u6.view(-1, PATCH_DIM)).view_as(noisy_p_u6)
            assert den_p.shape == noisy_p_u6.shape
            epqd_mses[name] = compute_mse(den_p, clean_p_u6)
            assert epqd_mses[name] < noisy_p_mse
    ut07_res = "Noisy patch MSE={:.5f} -> ".format(noisy_p_mse) + ", ".join(
        f"{k}={v:.5f}" for k, v in epqd_mses.items()
    )
    print(f"[UT07] Pass | {ut07_res}")
    results["unit"].append(("UT07", "Pass", ut07_res))

    # UT08: Hybrid Quantum-Classical Training (Loss & Parameter Update)
    train_epqd_m = EPQDDenoiser(name="test-train", n_qubits=4, n_layers=1, ansatz="basic", patch_dim=PATCH_DIM).to(DEVICE)
    train_epqd_m.train()
    q_before = train_epqd_m.q_weights.detach().clone()
    pre_before = train_epqd_m.pre_proj[0].weight.detach().clone()
    post_before = train_epqd_m.post_proj[0].weight.detach().clone()

    opt = optim.Adam(train_epqd_m.parameters(), lr=5e-3)
    crit = nn.MSELoss()
    x_tr = noisy_p_u6[:4].reshape(-1, PATCH_DIM)
    y_tr = clean_p_u6[:4].reshape(-1, PATCH_DIM)

    pred1 = train_epqd_m(x_tr)
    loss1 = crit(pred1, y_tr)
    opt.zero_grad()
    loss1.backward()
    q_grad_norm = float(train_epqd_m.q_weights.grad.norm().item())
    opt.step()

    pred2 = train_epqd_m(x_tr)
    loss2 = crit(pred2, y_tr)
    q_delta = float((train_epqd_m.q_weights.detach() - q_before).abs().max().item())
    pre_delta = float((train_epqd_m.pre_proj[0].weight.detach() - pre_before).abs().max().item())
    post_delta = float((train_epqd_m.post_proj[0].weight.detach() - post_before).abs().max().item())
    assert np.isfinite(loss1.item()) and q_grad_norm > 0 and q_delta > 0 and pre_delta > 0 and post_delta > 0
    ut08_res = (
        f"Loss {loss1.item():.5f}->{loss2.item():.5f}, q_grad_norm={q_grad_norm:.5f}, "
        f"max_delta(q={q_delta:.5f}, pre={pre_delta:.5f}, post={post_delta:.5f})"
    )
    print(f"[UT08] Pass | {ut08_res}")
    results["unit"].append(("UT08", "Pass", ut08_res))

    # UT09: Image Reconstruction & Quality Evaluation (MSE, PSNR, SSIM)
    with torch.no_grad():
        clean_r, noisy_r, epqd_r = epqd_full_reconstruction(epqd_models["EPQD-L"], ae, test_imgs)
    orig_01 = to_01(test_imgs)
    noisy_01 = to_01(noisy_r)
    epqd_01 = to_01(epqd_r)

    mse_n = compute_mse(noisy_01, orig_01)
    psnr_n = compute_psnr(noisy_01, orig_01)
    ssim_n = compute_ssim(noisy_01, orig_01)

    mse_e = compute_mse(epqd_01, orig_01)
    psnr_e = compute_psnr(epqd_01, orig_01)
    ssim_e = compute_ssim(epqd_01, orig_01)
    assert mse_e < mse_n and psnr_e > psnr_n and ssim_e > ssim_n
    ut09_res = (
        f"32 samples: Noisy(MSE={mse_n:.5f}, PSNR={psnr_n:.2f}dB, SSIM={ssim_n:.4f}) -> "
        f"EPQD-L(MSE={mse_e:.5f}, PSNR={psnr_e:.2f}dB, SSIM={ssim_e:.4f})"
    )
    print(f"[UT09] Pass | {ut09_res}")
    results["unit"].append(("UT09", "Pass", ut09_res))

    # ══════════════════════════════════════════════════════════════════════════
    # 2. INTEGRATION TESTING (IT01 - IT05)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 75)
    print("2. INTEGRATION TESTING")
    print("─" * 75)

    imgs_16 = test_imgs[:16]
    with torch.no_grad():
        # IT01: Module 1 + Module 2
        z_it1 = ae.encoder(imgs_16)
        p_it1, nh1, nw1 = extract_patches(z_it1, PATCH_SIZE, STRIDE)
        q_it1 = qpe(p_it1[:4])
        assert z_it1.shape == (16, 64, 7, 7) and p_it1.shape == (16, 36, 256) and q_it1.shape == (4, 36, 4)
        it01_res = f"M1 latent {tuple(z_it1.shape)} -> M2 patches {tuple(p_it1.shape)} -> PQC {tuple(q_it1.shape)}"
        print(f"[IT01] Pass | {it01_res}")
        results["integration"].append(("IT01", "Pass", it01_res))

        # IT02: Module 2 + Module 3
        stitched_4ch = patches_to_latent(q_it1, nh1, nw1, PATCH_SIZE, STRIDE, N_QUBITS)
        stitch_4to64 = StitchingCNN(in_channels=N_QUBITS, out_channels=LATENT_CHANNELS).to(DEVICE)
        refined_it2 = stitch_4to64(stitched_4ch)
        assert stitched_4ch.shape == (4, 4, 7, 7) and refined_it2.shape == (4, 64, 7, 7)
        it02_res = f"M2 PQC {tuple(q_it1.shape)} -> patches_to_latent {tuple(stitched_4ch.shape)} -> StitchingCNN {tuple(refined_it2.shape)}"
        print(f"[IT02] Pass | {it02_res}")
        results["integration"].append(("IT02", "Pass", it02_res))

        # IT03: Module 3 + Module 4
        noisy_z_it3 = z_it1 + torch.randn_like(z_it1) * NOISE_STDDEV
        p_it3, nh3, nw3 = extract_patches(noisy_z_it3, PATCH_SIZE, STRIDE)
        mlp_p_it3 = mlp(p_it3.view(-1, PATCH_DIM)).view_as(p_it3)
        mlp_z_it3 = fold_patches_spatial(mlp_p_it3, nh3, nw3, PATCH_SIZE, STRIDE, LATENT_CHANNELS)
        mlp_refined_it3 = stitch(mlp_z_it3)
        unet_refined_it3 = stitch(unet(noisy_z_it3))
        assert mlp_refined_it3.shape == (16, 64, 7, 7) and unet_refined_it3.shape == (16, 64, 7, 7)
        it03_res = (
            f"M4 MLP->fold->M3 Stitch {tuple(mlp_refined_it3.shape)}, MSE={compute_mse(mlp_refined_it3, z_it1):.5f}; "
            f"U-Net->M3 Stitch {tuple(unet_refined_it3.shape)}, MSE={compute_mse(unet_refined_it3, z_it1):.5f}"
        )
        print(f"[IT03] Pass | {it03_res}")
        results["integration"].append(("IT03", "Pass", it03_res))

        # IT04: Module 4 + Module 5
        epqd_p_it4 = epqd_models["EPQD-L"](p_it3.view(-1, PATCH_DIM)).view_as(p_it3)
        epqd_z_it4 = fold_patches_spatial(epqd_p_it4, nh3, nw3, PATCH_SIZE, STRIDE, LATENT_CHANNELS)
        mse_mlp_lat = compute_mse(mlp_z_it3, z_it1)
        mse_unet_lat = compute_mse(unet(noisy_z_it3), z_it1)
        mse_epqd_lat = compute_mse(epqd_z_it4, z_it1)
        assert epqd_z_it4.shape == unet(noisy_z_it3).shape == mlp_z_it3.shape == (16, 64, 7, 7)
        it04_res = f"Shared noisy latent {tuple(noisy_z_it3.shape)}: U-Net={mse_unet_lat:.5f}, MLP={mse_mlp_lat:.5f}, EPQD-L={mse_epqd_lat:.5f}"
        print(f"[IT04] Pass | {it04_res}")
        results["integration"].append(("IT04", "Pass", it04_res))

        # IT05: Module 1 + Module 5
        _, noisy_it5, epqd_it5 = epqd_full_reconstruction(epqd_models["EPQD-L"], ae, imgs_16)
        assert epqd_it5.shape == imgs_16.shape == (16, 1, 28, 28)
        psnr_it5 = compute_psnr(to_01(epqd_it5), to_01(imgs_16))
        ssim_it5 = compute_ssim(to_01(epqd_it5), to_01(imgs_16))
        it05_res = f"M1+M5 End-to-end recon {tuple(epqd_it5.shape)}, PSNR={psnr_it5:.2f}dB, SSIM={ssim_it5:.4f}"
        print(f"[IT05] Pass | {it05_res}")
        results["integration"].append(("IT05", "Pass", it05_res))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. SYSTEM TESTING (ST01 - ST05)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 75)
    print("3. SYSTEM TESTING")
    print("─" * 75)

    with torch.no_grad():
        # ST01: Complete System Execution (All 5 modules in sequence)
        z_st1 = ae.encoder(imgs_16)
        noisy_z_st1 = z_st1 + torch.randn_like(z_st1) * NOISE_STDDEV
        p_st1, nh_s1, nw_s1 = extract_patches(noisy_z_st1, PATCH_SIZE, STRIDE)
        epqd_p_st1 = epqd_models["EPQD-L"](p_st1.view(-1, PATCH_DIM)).view_as(p_st1)
        folded_st1 = fold_patches_spatial(epqd_p_st1, nh_s1, nw_s1, PATCH_SIZE, STRIDE, LATENT_CHANNELS)
        refined_st1 = stitch(folded_st1)
        recon_st1 = ae.decoder(refined_st1)
        assert recon_st1.shape == (16, 1, 28, 28)
        mse_st1 = compute_mse(to_01(recon_st1), to_01(imgs_16))
        psnr_st1 = compute_psnr(to_01(recon_st1), to_01(imgs_16))
        ssim_st1 = compute_ssim(to_01(recon_st1), to_01(imgs_16))
        st01_res = f"Full 5-module sequence executed; Recon {tuple(recon_st1.shape)}, MSE={mse_st1:.5f}, PSNR={psnr_st1:.2f}dB, SSIM={ssim_st1:.4f}"
        print(f"[ST01] Pass | {st01_res}")
        results["system"].append(("ST01", "Pass", st01_res))

        # ST02: Quantum-Enhanced Diffusion Across All 4 Quantum Configs
        cfg_metrics = []
        for cname, em in epqd_models.items():
            _, _, er = epqd_full_reconstruction(em, ae, imgs_16)
            m_val = compute_mse(to_01(er), to_01(imgs_16))
            p_val = compute_psnr(to_01(er), to_01(imgs_16))
            s_val = compute_ssim(to_01(er), to_01(imgs_16))
            cfg_metrics.append(f"{cname}(MSE={m_val:.5f}, PSNR={p_val:.2f}dB, SSIM={s_val:.4f})")
        st02_res = "; ".join(cfg_metrics)
        print(f"[ST02] Pass | {st02_res}")
        results["system"].append(("ST02", "Pass", st02_res))

    # ST03: Hybrid Quantum-Classical Training (2-epoch mini training run)
    st3_model = EPQDDenoiser(name="st3-epqd", n_qubits=4, n_layers=1, ansatz="basic", patch_dim=PATCH_DIM).to(DEVICE)
    st3_opt = optim.Adam(st3_model.parameters(), lr=5e-3)
    st3_crit = nn.MSELoss()
    st3_q_init = st3_model.q_weights.detach().clone()
    st3_x = noisy_p_u6[:8].reshape(-1, PATCH_DIM)
    st3_y = clean_p_u6[:8].reshape(-1, PATCH_DIM)
    st3_losses = []
    st3_model.train()
    for ep in range(2):
        pred = st3_model(st3_x)
        loss = st3_crit(pred, st3_y)
        st3_opt.zero_grad()
        loss.backward()
        st3_opt.step()
        st3_losses.append(float(loss.item()))
    st3_q_diff = float((st3_model.q_weights.detach() - st3_q_init).abs().max().item())
    assert st3_losses[-1] < st3_losses[0] and st3_q_diff > 0
    st03_res = f"2-epoch hybrid training on 288 patches: Loss {st3_losses[0]:.5f}->{st3_losses[1]:.5f}, q_weights max_delta={st3_q_diff:.5f}"
    print(f"[ST03] Pass | {st03_res}")
    results["system"].append(("ST03", "Pass", st03_res))

    # ST04: Unconditional Image Generation from Pure Gaussian Noise N(0, 1)
    with torch.no_grad():
        pure_noise_z = torch.randn(8, 64, 7, 7, device=DEVICE)
        pn_patches, nh_pn, nw_pn = extract_patches(pure_noise_z, PATCH_SIZE, STRIDE)
        pn_den = epqd_models["EPQD-L"](pn_patches.view(-1, PATCH_DIM)).view_as(pn_patches)
        pn_folded = fold_patches_spatial(pn_den, nh_pn, nw_pn, PATCH_SIZE, STRIDE, LATENT_CHANNELS)
        pn_gen = ae.decoder(pn_folded)
        pn_std = float(to_01(pn_gen).std().item())
        assert pn_gen.shape == (8, 1, 28, 28) and torch.isfinite(pn_gen).all()
    st04_res = (
        f"Generated {tuple(pn_gen.shape)} from N(0,1) noise (pixel std={pn_std:.4f}); "
        f"model is trained as single-step sigma=0.3 denoiser rather than multi-step DDPM from pure noise"
    )
    print(f"[ST04] Partially Verified | {st04_res}")
    results["system"].append(("ST04", "Partially Verified", st04_res))

    # ST05: System-Wide Model Comparison & Evaluation (32 test samples)
    with torch.no_grad():
        z_32 = ae.encoder(test_imgs)
        noisy_z_32 = z_32 + torch.randn_like(z_32) * NOISE_STDDEV
        p_32, nh32, nw32 = extract_patches(noisy_z_32, PATCH_SIZE, STRIDE)
        o32 = to_01(test_imgs)

        eval_rows = []
        # Conv-AE Clean
        c_32 = to_01(ae.decoder(z_32))
        eval_rows.append(("Conv-AE Clean", compute_mse(c_32, o32), compute_psnr(c_32, o32), compute_ssim(c_32, o32)))
        # Noisy
        n_32 = to_01(ae.decoder(noisy_z_32))
        eval_rows.append(("Noisy (sigma=0.3)", compute_mse(n_32, o32), compute_psnr(n_32, o32), compute_ssim(n_32, o32)))
        # MLP
        mlp_p32 = mlp(p_32.view(-1, PATCH_DIM)).view_as(p_32)
        mlp_z32 = fold_patches_spatial(mlp_p32, nh32, nw32, PATCH_SIZE, STRIDE, LATENT_CHANNELS)
        m_32 = to_01(ae.decoder(mlp_z32))
        eval_rows.append(("MLP", compute_mse(m_32, o32), compute_psnr(m_32, o32), compute_ssim(m_32, o32)))
        # U-Net
        u_32 = to_01(ae.decoder(unet(noisy_z_32)))
        eval_rows.append(("U-Net", compute_mse(u_32, o32), compute_psnr(u_32, o32), compute_ssim(u_32, o32)))
        # EPQD configs
        for cname, em in epqd_models.items():
            _, _, er32 = epqd_full_reconstruction(em, ae, test_imgs)
            e_32 = to_01(er32)
            eval_rows.append((cname, compute_mse(e_32, o32), compute_psnr(e_32, o32), compute_ssim(e_32, o32)))

    print("\n  [ST05 Image-Space Evaluation on 32 Test Samples (0..1 scale)]")
    print(f"  {'Model':<20} {'MSE':>10} {'PSNR (dB)':>12} {'SSIM':>10}")
    print("  " + "-" * 54)
    for mname, m_mse, m_psnr, m_ssim in eval_rows:
        print(f"  {mname:<20} {m_mse:10.6f} {m_psnr:12.2f} {m_ssim:10.4f}")

    st05_res = (
        f"32 samples evaluated: U-Net(MSE={eval_rows[3][1]:.5f}, PSNR={eval_rows[3][2]:.2f}dB, SSIM={eval_rows[3][3]:.4f}), "
        f"MLP(MSE={eval_rows[2][1]:.5f}, PSNR={eval_rows[2][2]:.2f}dB, SSIM={eval_rows[2][3]:.4f}), "
        f"EPQD-L(MSE={eval_rows[6][1]:.5f}, PSNR={eval_rows[6][2]:.2f}dB, SSIM={eval_rows[6][3]:.4f})"
    )
    print(f"[ST05] Pass | {st05_res}")
    results["system"].append(("ST05", "Pass", st05_res))

    # Summary Counts
    all_tests = results["unit"] + results["integration"] + results["system"]
    n_pass = sum(1 for _, s, _ in all_tests if s == "Pass")
    n_pv = sum(1 for _, s, _ in all_tests if s == "Partially Verified")
    n_fail = sum(1 for _, s, _ in all_tests if s == "Fail")

    print("\n" + "=" * 75)
    print(f"VALIDATION SUMMARY: Total={len(all_tests)} | Pass={n_pass} | Partially Verified={n_pv} | Fail={n_fail}")
    print(f"Elapsed time: {time.time() - t_start:.2f}s")
    print("=" * 75)

    os.makedirs("outputs/summary", exist_ok=True)
    with open("outputs/summary/validation_results.json", "w") as f:
        json.dump({"results": results, "eval_rows": eval_rows}, f, indent=2)


if __name__ == "__main__":
    main()
