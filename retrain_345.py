"""
Retrain Modules 3, 4, 5 after Module 1 (Conv-AE) is properly trained.
Run AFTER module1_autoencoder.py completes.

Usage:
    python retrain_345.py
"""
import os, time
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def banner(msg):
    print("\n" + "█" * 60)
    print(f"  {msg}")
    print("█" * 60)

# ── Verify AE is trained ──────────────────────────────────────
assert os.path.exists("models/conv_ae.pth"), "Run module1_autoencoder.py first!"
from module1_autoencoder import ConvAutoencoder
ae = ConvAutoencoder().to(DEVICE)
ae.load_state_dict(torch.load("models/conv_ae.pth", map_location=DEVICE))
ae.eval()

# Quick sanity: check decoder output is not all-gray
import torch
with torch.no_grad():
    dummy_z = torch.randn(1, 64, 7, 7).to(DEVICE)
    out = ae.decoder(dummy_z)
std_val = out.std().item()
print(f"\n  AE decoder output std (should be > 0.1): {std_val:.4f}")
if std_val < 0.05:
    print("  ⚠️  WARNING: AE may not be fully trained. Consider running module1 more epochs.")
else:
    print("  ✓  AE looks trained — proceeding with Modules 3, 4, 5\n")

# ── Module 3 ──────────────────────────────────────────────────
banner("MODULE 3 – Classical Stitching CNN (re-train)")
t0 = time.time()
import module3_stitching_cnn as m3
m3.main()
print(f"  Module 3 done in {time.time()-t0:.0f}s")

# ── Module 4 ──────────────────────────────────────────────────
banner("MODULE 4 – Classical Baselines (re-train)")
t0 = time.time()
import module4_classical_baselines as m4
m4.main()
print(f"  Module 4 done in {time.time()-t0:.0f}s")

# ── Module 5 ──────────────────────────────────────────────────
banner("MODULE 5 – EPQD (re-train + generate final outputs)")
t0 = time.time()
import module5_epqd as m5
m5.main()
print(f"  Module 5 done in {time.time()-t0:.0f}s")

print("\n✅  PHQD COMPLETE — check outputs/module5/PHQD_full_comparison.png")
