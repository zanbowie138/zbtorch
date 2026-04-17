"""Shared utilities for zbtorch training scripts."""

import time
import numpy as np
import matplotlib.pyplot as plt


# ── Plot theme ─────────────────────────────────────────────────────────────────
DARK_BG  = "#16213e"
GRID_COL = "#0f3460"
TEXT_COL = "#e0e0e0"
ACCENT   = "#e94560"
VAL_COL  = "#f5a623"


def style_ax(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=12, pad=8)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.6)


# ── Backend-agnostic helpers ───────────────────────────────────────────────────
def scalar(t):
    """Extract a Python float from a Tensor (works for both backends)."""
    d = t.data
    return d[0] if isinstance(d, list) else float(d)


def sgd_step(model, lr):
    """In-place SGD update compatible with both backends.

    The C++ backend stores data as Python lists; the pure-Python backend stores
    data as numpy arrays.  We preserve the original storage type so downstream
    code doesn't break.
    """
    for p in model.parameters():
        updated = np.asarray(p.data) - lr * np.asarray(p.grad)
        p.data = updated.tolist() if isinstance(p.data, list) else updated


# ── Training utilities ─────────────────────────────────────────────────────────
def train_loop(model, xs_train, ys_train, steps, lr, Tensor,
               xs_val=None, ys_val=None, normalize=False,
               log_every=0, on_step=None):
    """Run a timed SGD training loop.

    Parameters
    ----------
    normalize : bool
        Divide loss by the number of training samples (useful for larger
        datasets where the raw sum would dwarf the learning rate).
    log_every : int
        Print train/val loss every this many steps (0 = silent).
    on_step : callable(step, preds, train_loss) or None
        Called at the end of each step with the forward-pass predictions and
        the scalar train loss for that step.

    Returns
    -------
    train_loss_history, val_loss_history, step_times, forward_times, backward_times
    """
    train_loss_history = []
    val_loss_history   = []
    step_times         = []
    forward_times      = []
    backward_times     = []

    for step in range(steps):
        t_step = time.perf_counter()

        t_fwd = time.perf_counter()
        preds = [model(x)[0] for x in xs_train]
        loss  = sum(((p - Tensor(y)) ** 2 for p, y in zip(preds, ys_train)),
                    Tensor(0.0))
        if normalize:
            loss = loss * (1.0 / len(xs_train))
        forward_times.append(time.perf_counter() - t_fwd)

        model.zero_grad()

        t_bwd = time.perf_counter()
        loss.backward(cache = True)
        backward_times.append(time.perf_counter() - t_bwd)

        sgd_step(model, lr)
        step_times.append(time.perf_counter() - t_step)

        train_loss = scalar(loss)
        train_loss_history.append(train_loss)

        if xs_val is not None:
            val_loss = eval_mse(model, xs_val, ys_val)
            val_loss_history.append(val_loss)
            if log_every and (step + 1) % log_every == 0:
                print(f"  step {step+1:>3}/{steps}  "
                      f"train={train_loss:.4f}  val={val_loss:.4f}", flush=True)

        if on_step is not None:
            on_step(step, preds, train_loss)

    return train_loss_history, val_loss_history, step_times, forward_times, backward_times


def eval_mse(model, xs, ys):
    """Compute MSE without building a gradient graph."""
    preds = np.array([scalar(model(x)[0]) for x in xs])
    return float(np.mean((preds - np.array(ys)) ** 2))


def class_accuracy(model, xs, ys_list):
    """Return (class-0 acc, class-1 acc, overall acc) in percent."""
    preds   = np.array([scalar(model(x)[0]) for x in xs])
    targets = np.array(ys_list)
    c0 = int(np.sum(preds[targets == 0] < 0.5))
    t0 = int(np.sum(targets == 0))
    c1 = int(np.sum(preds[targets == 1] >= 0.5))
    t1 = int(np.sum(targets == 1))
    acc0    = c0 / t0 * 100 if t0 else 0.0
    acc1    = c1 / t1 * 100 if t1 else 0.0
    overall = (c0 + c1) / len(ys_list) * 100
    return acc0, acc1, overall


def build_grid(model, x_lo, x_hi, y_lo, y_hi, res=80):
    """Evaluate the model on a 2-D meshgrid for decision boundary plots."""
    gx, gy = np.meshgrid(np.linspace(x_lo, x_hi, res),
                         np.linspace(y_lo, y_hi, res))
    grid_preds = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            grid_preds[i, j] = scalar(model([gx[i, j], gy[i, j]])[0])
    return gx, gy, grid_preds


def print_benchmark(label, steps, step_times, forward_times, backward_times):
    total_ms = sum(step_times) * 1000
    avg_ms   = np.mean(step_times) * 1000
    fwd_ms   = np.mean(forward_times) * 1000
    bwd_ms   = np.mean(backward_times) * 1000
    print(f"\n{'='*52}")
    print(f"  Benchmark  ({label}, {steps} steps)")
    print(f"{'='*52}")
    print(f"  Total training time : {total_ms:>8.2f} ms")
    print(f"  Avg step time       : {avg_ms:>8.3f} ms")
    print(f"  Min / Max step      : "
          f"{min(step_times)*1000:>7.3f} / {max(step_times)*1000:.3f} ms")
    print(f"  Avg forward pass    : {fwd_ms:>8.3f} ms  ({fwd_ms/avg_ms*100:.1f}%)")
    print(f"  Avg backward pass   : {bwd_ms:>8.3f} ms  ({bwd_ms/avg_ms*100:.1f}%)")
    print(f"{'='*52}\n")
