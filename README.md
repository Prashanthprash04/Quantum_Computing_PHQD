# PHQD – Patch-based Hybrid Quantum Diffusion

A complete implementation of the **PHQD** method for high-quality image reconstruction using a hybrid classical-quantum pipeline on Fashion-MNIST.

---

## Project Structure

```
PHQD_MAJOR/
├── module1_autoencoder.py           # M1: Classical Conv-AE
├── module2_quantum_patch_encoding.py# M2: Quantum Patch Encoding
├── module3_stitching_cnn.py         # M3: Classical Stitching CNN
├── module4_classical_baselines.py   # M4: MLP + U-Net Baselines
├── module5_epqd.py                  # M5: EPQD Quantum Denoiser
├── run_phqd.py                      # Main pipeline runner
├── test_phqd.py                     # Smoke test (all modules)
├── requirements.txt
├── models/                          # Saved model weights (.pth)
└── outputs/
    ├── module1/   # Reconstructions, loss curves
    ├── module2/   # Quantum expectation heatmaps, patch grid
    ├── module3/   # Stitching comparison, latent maps
    ├── module4/   # Baseline comparison, MSE chart
    ├── module5/   # EPQD reconstructions, full comparison
    └── summary/   # Final MSE chart, JSON report
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Full Pipeline (all 5 modules)
```bash
python run_phqd.py
```

### Single Module
```bash
python run_phqd.py --module 1   # Conv-AE only
python run_phqd.py --module 4   # Classical baselines only
python run_phqd.py --module 5   # EPQD only
```

### Quick Smoke Test (verify all imports & shapes)
```bash
python test_phqd.py
```

### Skip Quantum Modules (fast classical-only run)
```bash
python run_phqd.py --skip-quantum
```

### Custom Epochs
```bash
python run_phqd.py --epochs-ae 30 --epochs-epqd 20
```

---

## Module Overview

| # | Module | Input | Output | Key Method |
|---|--------|-------|--------|------------|
| 1 | Conv-AE | 1×28×28 image | 64×7×7 latent | Encoder/Decoder CNN |
| 2 | Quantum Patch Encoding | 64×7×7 latent | (NP, N_qubits) | PQC w/ angle encoding |
| 3 | Stitching CNN | (NP, 4) patches | 64×7×7 refined latent | Overlap-add + refinement |
| 4 | Classical Baselines | noisy patches | clean patches | MLP, U-Net |
| 5 | EPQD | noisy patches | clean patches | PQC + pre/post MLP |

---

## EPQD Configurations

| Config | Qubits | Layers | Ansatz |
|--------|--------|--------|--------|
| EPQD-S | 4 | 1 | Basic (CNOT ring + Rot) |
| EPQD-M | 4 | 2 | Basic |
| EPQD-L | 4 | 3 | Basic |
| EPQD-IQP | 4 | 2 | IQP (ZZ interactions) |

---

## Output Images

All outputs are saved at **300 DPI** (high quality):

- `outputs/module1/reconstruction.png` – Original vs AE reconstruction
- `outputs/module4/baseline_comparison.png` – 5-row comparison grid
- `outputs/module5/PHQD_full_comparison.png` – **Master comparison (all models)**
- `outputs/module5/PHQD_MSE_comparison.png` – MSE bar chart
- `outputs/module5/pqc_circuit_diagram.png` – PQC circuit diagram
- `outputs/summary/PHQD_final_MSE.png` – Final summary MSE chart

---

## Architecture Diagram

```
Fashion-MNIST Image (1×28×28)
         │
    ┌────▼────┐
    │ Encoder │  (Conv-AE Module 1)
    └────┬────┘
         │ Latent (64×7×7)
    ┌────▼──────────────┐
    │  Patch Extraction  │  Overlapping 2×2 patches
    └────┬──────────────┘
         │ Patches (NP × 256)
    ┌────▼──────────────┐
    │  Add Gaussian Noise │  σ = 0.3  (training signal)
    └────┬──────────────┘
         │
    ┌────▼──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  EPQD (Quantum)    │  vs │  MLP Baseline │  vs │ U-Net Baseline│
    │  PQC Denoiser      │     └──────┬───────┘     └──────┬───────┘
    └────┬──────────────┘            │                     │
         │  Denoised Patches         ↓ MSE Comparison       ↓
    ┌────▼──────────────┐
    │  Stitch + Refine   │  (Module 3)
    └────┬──────────────┘
         │ Refined Latent (64×7×7)
    ┌────▼────┐
    │ Decoder │  (Conv-AE Module 1)
    └────┬────┘
         │
    High-Quality Reconstructed Image (1×28×28)
```
