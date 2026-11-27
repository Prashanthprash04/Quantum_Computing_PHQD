"""
Module 2: Quantum Patch Encoding
=================================
- Extracts overlapping patches from the Conv-AE latent representations (64×7×7)
- Encodes each patch's features into quantum-compatible angles
- Applies parameterized quantum circuits (PQC) to process each patch
- Returns quantum-processed patch embeddings for Module 3
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────── Configuration ───────────────────────────────────
PATCH_SIZE      = 2          # Spatial patch size (pixels)
STRIDE          = 1          # Stride for overlapping patches
N_QUBITS        = 4          # Qubits per patch circuit
N_LAYERS        = 2          # Variational layers in PQC
LATENT_CHANNELS = 64         # Must match Module 1
LATENT_HW       = 7          # Spatial size of latent map (7×7)
DEVICE_TORCH    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR        = "outputs/module2"
MODEL_PATH      = "models/quantum_encoder.pt"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ─────────────────────────── Patch Extraction ────────────────────────────────
def extract_patches(latent: torch.Tensor,
                    patch_size: int = PATCH_SIZE,
                    stride: int     = STRIDE) -> torch.Tensor:
    """
    Extract overlapping spatial patches from a latent feature map.

    Args:
        latent : (B, C, H, W) tensor
        patch_size : spatial size of each patch
        stride     : sliding-window stride

    Returns:
        patches : (B, n_patches, C * patch_size * patch_size)
    """
    B, C, H, W = latent.shape
    patches = latent.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # patches: (B, C, n_h, n_w, patch_size, patch_size)
    n_h = patches.shape[2]
    n_w = patches.shape[3]
    patches = patches.contiguous().view(B, C, n_h, n_w, patch_size * patch_size)
    # (B, C, n_h, n_w, p²) → (B, n_h, n_w, C, p²)
    patches = patches.permute(0, 2, 3, 1, 4)
    # → (B, n_h*n_w, C*p²)
    patches = patches.contiguous().view(B, n_h * n_w, C * patch_size * patch_size)
    return patches, n_h, n_w


def patches_to_angles(patches: torch.Tensor) -> torch.Tensor:
    """
    Map patch features to angles in [0, π] for quantum angle encoding.
    Uses tanh normalisation then scales to [0, π].
    """
    normed = torch.tanh(patches)                # → (-1, 1)
    angles = (normed + 1.0) * (torch.pi / 2.0) # → (0, π)
    return angles


# ─────────────────────────── Quantum Circuit ─────────────────────────────────
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_patch_circuit(angles, weights):
    """
    Parameterized Quantum Circuit for one patch.

    angles  : (N_QUBITS,) – angle-encoded input features
    weights : (N_LAYERS, N_QUBITS, 3) – trainable rotation params (Rx, Ry, Rz)

    Returns : list of N_QUBITS Pauli-Z expectation values in [-1, 1]
    """
    # ── Angle Encoding ──
    for i in range(N_QUBITS):
        qml.RY(angles[i], wires=i)

    # ── Variational Layers ──
    for layer in range(N_LAYERS):
        # Entanglement ring
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
        # Parameterised rotations
        for i in range(N_QUBITS):
            qml.Rot(weights[layer, i, 0],
                    weights[layer, i, 1],
                    weights[layer, i, 2], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QuantumPatchEncoder(nn.Module):
    """
    Classical projection → Quantum PQC → Classical projection
    Input  : (B, n_patches, C*p²)
    Output : (B, n_patches, N_QUBITS)
    """
    def __init__(self,
                 in_features : int  = LATENT_CHANNELS * PATCH_SIZE**2,
                 n_qubits    : int  = N_QUBITS,
                 n_layers    : int  = N_LAYERS):
        super().__init__()
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers

        # Dimensionality reduction before quantum circuit
        self.pre_proj = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.LayerNorm(32),
            nn.Tanh(),
            nn.Linear(32, n_qubits),
            nn.Tanh(),
        )

        # Trainable quantum weights
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        # Optional post-processing MLP
        self.post_proj = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Tanh(),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches : (B, n_patches, in_features)
        returns : (B, n_patches, n_qubits)
        """
        B, NP, _ = patches.shape
        # Classical projection to n_qubits angles
        angles = self.pre_proj(patches)              # (B, NP, n_qubits)
        # Map to [0, π]
        angles = (angles + 1.0) * (torch.pi / 2.0)

        # Apply quantum circuit to every patch
        q_out = []
        for b in range(B):
            patch_outs = []
            for p in range(NP):
                result = quantum_patch_circuit(
                    angles[b, p],      # (n_qubits,)
                    self.q_weights     # (n_layers, n_qubits, 3)
                )
                patch_outs.append(torch.stack(result))   # (n_qubits,)
            q_out.append(torch.stack(patch_outs))         # (NP, n_qubits)
        q_out = torch.stack(q_out).float()                 # (B, NP, n_qubits) – cast to float32

        return self.post_proj(q_out)


# ─────────────────────────── Training / Demo ─────────────────────────────────
def demo_quantum_encoding(latent_path="outputs/latents_train.npy",
                           n_samples=8, save_weights=True):
    """
    Demonstration: encode a small batch of latent maps through the quantum circuit
    and visualise the output expectation values as a heatmap.
    """
    print("\n[Quantum Patch Encoding Demo]")

    if not os.path.exists(latent_path):
        print(f"  Latent file not found: {latent_path}")
        print("  Generating random latents for demo …")
        latents = np.random.randn(n_samples, LATENT_CHANNELS, LATENT_HW, LATENT_HW).astype(np.float32)
    else:
        latents = np.load(latent_path)[:n_samples]

    latent_tensor = torch.tensor(latents, dtype=torch.float32)

    # Extract patches
    patches, n_h, n_w = extract_patches(latent_tensor, PATCH_SIZE, STRIDE)
    print(f"  Latent shape : {latent_tensor.shape}")
    print(f"  Patch grid   : {n_h}×{n_w} → {patches.shape[1]} patches each {patches.shape[2]}-dim")

    # Quantum encoder
    model = QuantumPatchEncoder(
        in_features=patches.shape[2],
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {n_params:,} (classical) + {N_LAYERS*N_QUBITS*3} (quantum weights)")

    # Forward pass (small batch for speed)
    with torch.no_grad():
        q_encoded = model(patches[:2])    # (2, NP, N_QUBITS)
    print(f"  Quantum encoded shape : {q_encoded.shape}")

    # Visualise expectation values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for idx, ax in enumerate(axes):
        ev = q_encoded[idx].numpy()          # (NP, N_QUBITS)
        im = ax.imshow(ev, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xlabel("Qubit"); ax.set_ylabel("Patch index")
        ax.set_title(f"Sample {idx} – PauliZ Expectation Values")
        plt.colorbar(im, ax=ax)
    plt.suptitle("Module 2 – Quantum Patch Encoding", fontsize=12)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "quantum_expectation_values.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    if save_weights:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  Model saved → {MODEL_PATH}")

    # Save encoded patches for downstream use
    with torch.no_grad():
        q_all = model(patches)
    np.save("outputs/quantum_encoded_patches.npy", q_all.numpy())
    print(f"  Quantum encoded patches saved → outputs/quantum_encoded_patches.npy")

    return model, q_encoded, (n_h, n_w)


def visualise_patch_grid(latent_tensor, n_h, n_w, sample_idx=0):
    """Visualise the spatial patch grid over the latent map (first channel)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(latent_tensor[sample_idx, 0].numpy(), cmap="viridis")
    # Draw patch boundaries
    for r in range(n_h + 1):
        ax.axhline(r * STRIDE - 0.5, color="red", linewidth=0.5, alpha=0.5)
    for c in range(n_w + 1):
        ax.axvline(c * STRIDE - 0.5, color="red", linewidth=0.5, alpha=0.5)
    ax.set_title(f"Module 2 – Patch Grid on Latent Map (sample {sample_idx}, ch 0)")
    out = os.path.join(SAVE_DIR, "patch_grid.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────── Entry Point ─────────────────────────────────────
def main():
    print("=" * 60)
    print("MODULE 2 – Quantum Patch Encoding")
    print("=" * 60)

    model, q_encoded, (n_h, n_w) = demo_quantum_encoding()

    latent_path = "outputs/latents_train.npy"
    if os.path.exists(latent_path):
        latents = torch.tensor(np.load(latent_path)[:4], dtype=torch.float32)
        visualise_patch_grid(latents, n_h, n_w)

    print("\nMODULE 2 COMPLETE ✓")
    return model


if __name__ == "__main__":
    main()
