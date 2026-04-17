import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from zbtorch import Tensor, MLP
from test_utils import (DARK_BG, GRID_COL, TEXT_COL, ACCENT, VAL_COL,
                        style_ax, build_grid, class_accuracy,
                        print_benchmark, train_loop)

_DIR = Path(__file__).parent


# ── Spiral dataset ─────────────────────────────────────────────────────────────
np.random.seed(42)

N     = 50
theta = np.linspace(0, 4 * np.pi, N)
r     = np.linspace(0.1, 1.0, N)
noise = 0.05

x0 = r * np.cos(theta)         + np.random.randn(N) * noise
y0 = r * np.sin(theta)         + np.random.randn(N) * noise
x1 = r * np.cos(theta + np.pi) + np.random.randn(N) * noise
y1 = r * np.sin(theta + np.pi) + np.random.randn(N) * noise

xs_np = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
ys_np = np.array([0.0] * N + [1.0] * N)

idx = np.random.permutation(2 * N)
xs_np, ys_np = xs_np[idx], ys_np[idx]

# ── Train / val split (80 / 20) ────────────────────────────────────────────────
split = int(0.8 * len(xs_np))
xs_train_np, ys_train_np = xs_np[:split], ys_np[:split]
xs_val_np,   ys_val_np   = xs_np[split:], ys_np[split:]

xs_train = xs_train_np.tolist()
ys_train = ys_train_np.tolist()
xs_val   = xs_val_np.tolist()
ys_val   = ys_val_np.tolist()

print(f"Train: {len(xs_train)} samples  |  Val: {len(xs_val)} samples")


np.random.seed(42)
model = MLP(2, [20, 20, 20, 1])

STEPS = 500
LR    = 0.1

train_loss_history, val_loss_history, step_times, fwd_times, bwd_times = train_loop(
    model, xs_train, ys_train, STEPS, LR, Tensor,
    xs_val=xs_val, ys_val=ys_val, normalize=True, log_every=50,
)

print_benchmark("zbtorch C++ backend", STEPS, step_times, fwd_times, bwd_times)


# ── Build decision boundary grid ──────────────────────────────────────────────
pad  = 0.25
gx, gy, grid_preds = build_grid(
    model,
    xs_np[:, 0].min() - pad, xs_np[:, 0].max() + pad,
    xs_np[:, 1].min() - pad, xs_np[:, 1].max() + pad,
    res=80,
)

train_acc0, train_acc1, train_overall = class_accuracy(model, xs_train, ys_train)
val_acc0,   val_acc1,   val_overall   = class_accuracy(model, xs_val,   ys_val)


# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#1a1a2e")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

avg_ms = np.mean(step_times) * 1000


# ── 1. Train vs val loss curve ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Train vs Val Loss  (MSE)")
steps_range = range(STEPS)
ax1.plot(steps_range, train_loss_history, color=ACCENT,  linewidth=1.8, label="train")
ax1.plot(steps_range, val_loss_history,   color=VAL_COL, linewidth=1.8, label="val", linestyle="--")
ax1.fill_between(steps_range, train_loss_history, alpha=0.10, color=ACCENT)
ax1.fill_between(steps_range, val_loss_history,   alpha=0.10, color=VAL_COL)
ax1.set_xlabel("Step", color=TEXT_COL, fontsize=9)
ax1.set_ylabel("Loss", color=TEXT_COL, fontsize=9)
ax1.set_xlim(0, STEPS - 1)
ax1.legend(fontsize=8, framealpha=0.3, facecolor=DARK_BG, labelcolor=TEXT_COL)


# ── 2. Decision boundary ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Decision Boundary  (after training)")
im = ax2.contourf(gx, gy, grid_preds, levels=50, cmap="RdYlGn", vmin=0, vmax=1)
ax2.contour(gx, gy, grid_preds, levels=[0.5], colors="white", linewidths=1.2, linestyles="--")
cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cb.ax.tick_params(colors=TEXT_COL, labelsize=8)
cb.set_label("Output", color=TEXT_COL, fontsize=8)

train_colors = ["#c0392b" if t == 0 else "#27ae60" for t in ys_train_np]
val_colors   = ["#c0392b" if t == 0 else "#27ae60" for t in ys_val_np]
ax2.scatter(xs_train_np[:, 0], xs_train_np[:, 1], c=train_colors, s=40, zorder=5,
            edgecolors="white", linewidths=0.6, label="train")
ax2.scatter(xs_val_np[:, 0],   xs_val_np[:, 1],   c=val_colors,   s=55, zorder=6,
            edgecolors="white", linewidths=1.2, marker="s", label="val")
ax2.legend(fontsize=8, framealpha=0.3, facecolor=DARK_BG, labelcolor=TEXT_COL)
ax2.set_xlabel("x₀", color=TEXT_COL, fontsize=9)
ax2.set_ylabel("x₁", color=TEXT_COL, fontsize=9)


# ── 3. Per-step timing histogram ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Step Latency Distribution  (ms)")
step_times_ms = np.array(step_times) * 1000
ax3.hist(step_times_ms, bins=40, color=ACCENT, alpha=0.85, edgecolor=DARK_BG, linewidth=0.4)
ax3.axvline(avg_ms, color="white", linewidth=1.2, linestyle="--",
            label=f"mean {avg_ms:.2f} ms")
ax3.set_xlabel("Step time (ms)", color=TEXT_COL, fontsize=9)
ax3.set_ylabel("Count", color=TEXT_COL, fontsize=9)
ax3.legend(fontsize=8, framealpha=0.3, facecolor=DARK_BG, labelcolor=TEXT_COL)


# ── 4. Train vs val accuracy ───────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Train vs Val Accuracy by Class")

categories  = ["Class 0", "Class 1", "Overall"]
train_accs  = [train_acc0, train_acc1, train_overall]
val_accs    = [val_acc0,   val_acc1,   val_overall]

x      = np.arange(len(categories))
width  = 0.35
bars_t = ax4.bar(x - width / 2, train_accs, width, label="train",
                 color=ACCENT,  alpha=0.85, edgecolor="white", linewidth=0.6)
bars_v = ax4.bar(x + width / 2, val_accs,   width, label="val",
                 color=VAL_COL, alpha=0.85, edgecolor="white", linewidth=0.6)

for bar, val in zip(bars_t, train_accs):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f"{val:.1f}%", ha="center", color=TEXT_COL, fontsize=8)
for bar, val in zip(bars_v, val_accs):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f"{val:.1f}%", ha="center", color=TEXT_COL, fontsize=8)

ax4.set_xticks(x)
ax4.set_xticklabels(categories, color=TEXT_COL, fontsize=9)
ax4.set_ylim(0, 120)
ax4.set_ylabel("Accuracy (%)", color=TEXT_COL, fontsize=9)
ax4.legend(fontsize=8, framealpha=0.3, facecolor=DARK_BG, labelcolor=TEXT_COL)

fig.suptitle("Spiral  ·  MLP 2→20→20→20→20→1  ·  MSE + SGD  ·  C++ backend",
             color=TEXT_COL, fontsize=13, y=0.98)

# plt.savefig(_DIR / "spiral_training.png", dpi=150, bbox_inches="tight",
#             facecolor=fig.get_facecolor())
plt.show()
