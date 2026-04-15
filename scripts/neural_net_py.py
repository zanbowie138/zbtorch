import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from zbtorch_py import Tensor, MLP

_DIR = Path(__file__).parent


# XOR dataset
xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [0.0, 1.0, 1.0, 0.0]
labels = ["(0,0)→0", "(0,1)→1", "(1,0)→1", "(1,1)→0"]
colors = ["#e74c3c", "#2ecc71", "#2ecc71", "#e74c3c"]  # red=0, green=1

np.random.seed(42)
model = MLP(2, [4, 4, 1])

STEPS = 200
LR = 0.1

loss_history = []
pred_history = [[] for _ in range(len(xs))]

step_times = []
forward_times = []
backward_times = []

for step in range(STEPS):
    t_step = time.perf_counter()

    t_fwd = time.perf_counter()
    preds = [model(x)[0] for x in xs]
    loss = sum(((p - Tensor(y)) ** 2 for p, y in zip(preds, ys)), Tensor(0.0))
    forward_times.append(time.perf_counter() - t_fwd)

    model.zero_grad()

    t_bwd = time.perf_counter()
    loss.backward()
    backward_times.append(time.perf_counter() - t_bwd)

    # zbtorch_py: p.data is numpy — no .tolist() conversion needed
    for p in model.parameters():
        p.data = p.data - LR * np.asarray(p.grad)

    step_times.append(time.perf_counter() - t_step)

    # zbtorch_py: p.data is a 0-d numpy array — use float() to extract scalar
    loss_history.append(float(loss.data))
    for i, pred in enumerate(preds):
        pred_history[i].append(float(pred.data))

total_ms = sum(step_times) * 1000
avg_ms   = np.mean(step_times) * 1000
fwd_ms   = np.mean(forward_times) * 1000
bwd_ms   = np.mean(backward_times) * 1000
print(f"\n{'='*52}")
print(f"  Benchmark  (zbtorch_py pure Python, {STEPS} steps)")
print(f"{'='*52}")
print(f"  Total training time : {total_ms:>8.2f} ms")
print(f"  Avg step time       : {avg_ms:>8.3f} ms")
print(f"  Min / Max step      : {min(step_times)*1000:>7.3f} / {max(step_times)*1000:.3f} ms")
print(f"  Avg forward pass    : {fwd_ms:>8.3f} ms  ({fwd_ms/avg_ms*100:.1f}%)")
print(f"  Avg backward pass   : {bwd_ms:>8.3f} ms  ({bwd_ms/avg_ms*100:.1f}%)")
print(f"{'='*52}\n")


# ── Build decision boundary grid ──────────────────────────────────────────────
res = 200
gx, gy = np.meshgrid(np.linspace(-0.3, 1.3, res), np.linspace(-0.3, 1.3, res))
grid_preds = np.zeros((res, res))
for i in range(res):
    for j in range(res):
        val = model([gx[i, j], gy[i, j]])[0]
        grid_preds[i, j] = float(val.data)


# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#1a1a2e")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

DARK_BG  = "#16213e"
GRID_COL = "#0f3460"
TEXT_COL = "#e0e0e0"
ACCENT   = "#e94560"

def style_ax(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=12, pad=8)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.6)


# ── 1. Loss curve ──────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Training Loss  (MSE)")
ax1.plot(loss_history, color=ACCENT, linewidth=1.8, label="loss")
ax1.set_xlabel("Step", color=TEXT_COL, fontsize=9)
ax1.set_ylabel("Loss", color=TEXT_COL, fontsize=9)
ax1.fill_between(range(STEPS), loss_history, alpha=0.15, color=ACCENT)
ax1.set_xlim(0, STEPS - 1)


# ── 2. Decision boundary ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Decision Boundary  (after training)")
im = ax2.contourf(gx, gy, grid_preds, levels=50, cmap="RdYlGn", vmin=0, vmax=1)
ax2.contour(gx, gy, grid_preds, levels=[0.5], colors="white", linewidths=1.2, linestyles="--")
cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cb.ax.tick_params(colors=TEXT_COL, labelsize=8)
cb.set_label("Output", color=TEXT_COL, fontsize=8)

xor_x = [p[0] for p in xs]
xor_y_coord = [p[1] for p in xs]
xor_targets = ys
point_colors = ["#c0392b" if t == 0 else "#27ae60" for t in xor_targets]
ax2.scatter(xor_x, xor_y_coord, c=point_colors, s=120, zorder=5,
            edgecolors="white", linewidths=1.2)
for (px, py), lbl in zip(xs, labels):
    ax2.text(px + 0.05, py + 0.05, lbl, color="white", fontsize=8,
             bbox=dict(facecolor="#00000066", edgecolor="none", pad=1.5))
ax2.set_xlabel("x₀", color=TEXT_COL, fontsize=9)
ax2.set_ylabel("x₁", color=TEXT_COL, fontsize=9)


# ── 3. Per-sample prediction trajectories ─────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Prediction Trajectories per Sample")
traj_colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
for i, (hist, lbl, c) in enumerate(zip(pred_history, labels, traj_colors)):
    ax3.plot(hist, color=c, linewidth=1.6, label=lbl)
    # dashed target line
    ax3.axhline(ys[i], color=c, linewidth=0.7, linestyle=":", alpha=0.5)
ax3.set_xlabel("Step", color=TEXT_COL, fontsize=9)
ax3.set_ylabel("Predicted value", color=TEXT_COL, fontsize=9)
ax3.set_xlim(0, STEPS - 1)
ax3.set_ylim(-0.1, 1.1)
legend = ax3.legend(fontsize=8, framealpha=0.3, facecolor=DARK_BG,
                    labelcolor=TEXT_COL, loc="center right")


# ── 4. Final predictions vs targets ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Final Predictions vs Targets")
final_preds = [pred_history[i][-1] for i in range(len(xs))]
x_pos = np.arange(len(xs))
bar_w = 0.35
ax4.bar(x_pos - bar_w / 2, ys, bar_w, label="Target",
        color="#3498db", alpha=0.85, edgecolor="white", linewidth=0.6)
ax4.bar(x_pos + bar_w / 2, final_preds, bar_w, label="Predicted",
        color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.6)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f"({x[0]:.0f},{x[1]:.0f})" for x in xs], color=TEXT_COL, fontsize=9)
ax4.set_ylim(0, 1.25)
ax4.set_ylabel("Value", color=TEXT_COL, fontsize=9)
for i, (t, p) in enumerate(zip(ys, final_preds)):
    ax4.text(i + bar_w / 2, p + 0.03, f"{p:.2f}", ha="center",
             color=TEXT_COL, fontsize=8)
legend4 = ax4.legend(fontsize=8, framealpha=0.3, facecolor=DARK_BG, labelcolor=TEXT_COL)

fig.suptitle("XOR  ·  MLP 2→4→4→1  ·  tanh activations  ·  MSE + SGD  ·  pure Python",
             color=TEXT_COL, fontsize=13, y=0.98)

plt.savefig(_DIR / "neural_net_py_training.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
