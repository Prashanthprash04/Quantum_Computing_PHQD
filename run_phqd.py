"""
PHQD Main Runner
=================
Orchestrates all 5 modules sequentially and produces a final
MSE comparison report + master summary figure.

Usage:
    python run_phqd.py                  # run all modules
    python run_phqd.py --module 1       # run a single module
    python run_phqd.py --skip-quantum   # skip quantum modules (for quick baseline test)
"""

import argparse
import os
import sys
import time
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────── Argument Parsing ────────────────────────────────
parser = argparse.ArgumentParser(description="PHQD Pipeline Runner")
parser.add_argument("--module",        type=int, default=0,
                    help="Run only this module (0 = all)")
parser.add_argument("--skip-quantum",  action="store_true",
                    help="Skip Modules 2, 3, 5 (quantum-heavy)")
parser.add_argument("--epochs-ae",     type=int, default=20)
parser.add_argument("--epochs-stitch", type=int, default=15)
parser.add_argument("--epochs-base",   type=int, default=20)
parser.add_argument("--epochs-epqd",   type=int, default=15)
args, _ = parser.parse_known_args()


# ─────────────────────────── Helper ──────────────────────────────────────────
def banner(title):
    print("\n" + "█" * 65)
    print(f"  {title}")
    print("█" * 65)


def timed(fn, *fn_args, **fn_kwargs):
    t0  = time.time()
    res = fn(*fn_args, **fn_kwargs)
    return res, time.time() - t0


# ─────────────────────────── Module 1 ────────────────────────────────────────
def run_module1(epochs):
    banner("MODULE 1 – Classical Convolutional Autoencoder (Conv-AE)")
    import module1_autoencoder as m1
    m1.EPOCHS = epochs
    return m1.main()


# ─────────────────────────── Module 2 ────────────────────────────────────────
def run_module2():
    banner("MODULE 2 – Quantum Patch Encoding")
    import module2_quantum_patch_encoding as m2
    return m2.main()


# ─────────────────────────── Module 3 ────────────────────────────────────────
def run_module3(epochs):
    banner("MODULE 3 – Classical Stitching CNN")
    import module3_stitching_cnn as m3
    m3.EPOCHS_STITCH = epochs
    return m3.main()


# ─────────────────────────── Module 4 ────────────────────────────────────────
def run_module4(epochs):
    banner("MODULE 4 – Classical Diffusion Baselines (MLP + U-Net)")
    import module4_classical_baselines as m4
    m4.EPOCHS = epochs
    return m4.main()


# ─────────────────────────── Module 5 ────────────────────────────────────────
def run_module5(epochs):
    banner("MODULE 5 – EPQD: Enhanced Parameterized Quantum Diffusion")
    import module5_epqd as m5
    m5.EPQD_EPOCHS = epochs
    return m5.main()


# ─────────────────────────── Summary Figure ──────────────────────────────────
def generate_summary_report(timings: dict, mse_results: dict):
    os.makedirs("outputs/summary", exist_ok=True)

    # ── MSE bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    names  = list(mse_results.keys())
    values = list(mse_results.values())
    colors = ["steelblue" if "MLP" in n else
              "darkorange" if ("UNet" in n or "U-Net" in n) else
              "mediumseagreen"
              for n in names]
    bars = ax.bar(names, values, color=colors, edgecolor="k", linewidth=0.5)
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=9)
    ax.set_ylabel("Test MSE")
    ax.set_title("PHQD – Final MSE Comparison: Classical vs Quantum Models",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/summary/PHQD_final_MSE.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Timing table ──
    print("\n" + "=" * 55)
    print(f"  {'Module':<30} {'Time':>12}")
    print("-" * 55)
    total = 0
    for mod, t in timings.items():
        print(f"  {mod:<30} {t:10.1f}s")
        total += t
    print(f"  {'TOTAL':<30} {total:10.1f}s")
    print("=" * 55)

    # ── Save JSON report ──
    report = {"mse": mse_results, "timings_s": timings}
    with open("outputs/summary/phqd_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n  Report saved → outputs/summary/phqd_report.json")
    print("  MSE chart  → outputs/summary/PHQD_final_MSE.png")


# ─────────────────────────── Main Orchestrator ───────────────────────────────
def main():
    print("=" * 65)
    print("  PHQD – Patch-based Hybrid Quantum Diffusion Pipeline")
    print(f"  Device : {DEVICE}")
    print("=" * 65)

    timings    = {}
    mse_results = {}

    only = args.module
    skip_q = args.skip_quantum

    # ── Module 1 ──
    if only in (0, 1):
        result, t = timed(run_module1, args.epochs_ae)
        timings["M1 Conv-AE"] = t
        if isinstance(result, float):
            mse_results["Conv-AE"] = result

    # ── Module 2 ──
    if only in (0, 2) and not skip_q:
        _, t = timed(run_module2)
        timings["M2 Quantum Patch Enc"] = t

    # ── Module 3 ──
    if only in (0, 3) and not skip_q:
        result, t = timed(run_module3, args.epochs_stitch)
        timings["M3 Stitching CNN"] = t
        if isinstance(result, tuple):
            mse_results["Stitching-CNN"] = result[1]

    # ── Module 4 ──
    if only in (0, 4):
        result, t = timed(run_module4, args.epochs_base)
        timings["M4 MLP+UNet Baselines"] = t
        if isinstance(result, tuple) and len(result) == 3:
            _, _, mse_dict = result
            mse_results.update(mse_dict)

    # ── Module 5 ──
    if only in (0, 5) and not skip_q:
        result, t = timed(run_module5, args.epochs_epqd)
        timings["M5 EPQD"] = t
        if isinstance(result, tuple) and len(result) == 2:
            _, mse_dict = result
            mse_results.update(mse_dict)

    # ── Summary ──
    if only == 0:
        banner("PHQD PIPELINE COMPLETE – Summary")
        generate_summary_report(timings, mse_results)

    print("\n✅  All requested modules finished.\n")


if __name__ == "__main__":
    main()
