import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

C = {"data": "white", "cls": "white", "q": "white", "edge": "black"}

def box(x, y, w, h, text, kind, fs=6.6, hatch=None):
    hatch = "////" if kind == "q" else None
    ls = (0, (2, 1.5)) if kind == "data" else "-"
    lw = 1.6 if kind == "q" else 0.9
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0",
                       fc=C[kind], ec=C["edge"], lw=lw, ls=ls, hatch=hatch)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25,
            bbox=dict(fc="white", ec="none", pad=1.2) if kind == "q" else None)
    return (x, y, w, h)

def arrow(p, q, rad=0.0):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=C["edge"],
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)

# Row 1
y1, h = 29, 11
w = 15.5; gap = 4.4; x = 1.5
r1 = []
labels1 = [("Fashion-MNIST\nimage\n$1{\\times}28{\\times}28$", "data"),
           ("Conv-AE\nencoder", "cls"),
           ("Latent map $z$\n$64{\\times}7{\\times}7$", "data"),
           ("Add noise\n$\\tilde z=z+\\sigma\\epsilon$\n$\\sigma=0.3$", "cls"),
           ("Overlapping\n$2{\\times}2$ patches\n$36{\\times}256$", "cls")]
for i, (t, k) in enumerate(labels1):
    r1.append(box(x + i * (w + gap), y1, w, h, t, k))
for i in range(4):
    a, b = r1[i], r1[i + 1]
    arrow((a[0] + a[2] + 0.3, y1 + h / 2), (b[0] - 0.3, y1 + h / 2))

# Row 2
y2 = 4; w2 = 11.8; gap2 = 2.4; x2 = 1.2
labels2 = [("Pre-projection\n$256\\rightarrow n$\nangles", "cls"),
           ("PQC\n$R_Y$, CNOT ring,\nRot, $\\langle Z\\rangle$", "q"),
           ("Post-projection\n$+$ residual\npath", "cls"),
           ("Fold-back\n(average\noverlaps)", "cls"),
           ("StitchingCNN\n(optional)", "cls"),
           ("Conv-AE\ndecoder", "cls"),
           ("Denoised\nimage", "data")]
r2 = []
for i, (t, k) in enumerate(labels2):
    r2.append(box(x2 + i * (w2 + gap2), y2, w2, h, t, k, fs=6.2))
for i in range(6):
    a, b = r2[i], r2[i + 1]
    arrow((a[0] + a[2] + 0.3, y2 + h / 2), (b[0] - 0.3, y2 + h / 2))

# group frame around EPQD denoiser
gx0 = r2[0][0] - 0.8; gx1 = r2[2][0] + r2[2][2] + 0.8
frame = FancyBboxPatch((gx0, y2 - 1.8), gx1 - gx0, h + 6.4, boxstyle="round,pad=0.2,rounding_size=1.2",
                       fc="none", ec=C["edge"], lw=0.8, ls=(0, (4, 3)))
ax.add_patch(frame)
ax.text((gx0 + gx1) / 2, y2 + h + 3.0, "Hybrid denoiser (EPQD), applied to every patch", ha="center",
        va="center", fontsize=6.4, style="italic")

# connector row1 -> row2 (elbow)
cx = last_x = r1[-1][0] + r1[-1][2] / 2
tx = r2[0][0] + r2[0][2] / 2
frame_top = y2 - 1.8 + h + 6.4
ax.plot([cx, cx], [y1 - 0.3, 24.6], color=C["edge"], lw=0.9)
ax.plot([cx, tx], [24.6, 24.6], color=C["edge"], lw=0.9)
arrow((tx, 24.6), (tx, frame_top + 0.3))
ax.text(cx - 1.5, 23.2, "MLP / U-Net baselines replace the hybrid denoiser", ha="right", va="center", fontsize=6.0, style="italic")

# legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="white", ec="black", ls=(0, (2, 1.5)), lw=0.9, label="Data"),
                   Patch(fc="white", ec="black", lw=0.9, label="Classical"),
                   Patch(fc="white", ec="black", lw=1.6, hatch="////", label="Quantum circuit")],
          loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=6.2, frameon=False)
fig.savefig("phqd_architecture_bw.png", dpi=300, bbox_inches="tight", facecolor="white")
