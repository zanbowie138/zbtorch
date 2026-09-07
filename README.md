# zbtorch

A small tensor library with a from-scratch autograd engine, built for learning how deep learning frameworks work under the hood.

The core (tensors, ops, autograd) is implemented in C++ with CPU and CUDA backends, and exposed to Python through pybind11. A simple `Neuron`/`Layer`/`MLP` API is built on top.

## Features

- Reverse-mode automatic differentiation over a dynamic computation graph
- CPU backend (OpenBLAS) and CUDA backend (custom kernels), selected per-tensor via `device="cpu"` / `device="cuda"`
- NumPy-compatible tensor construction and data access
- Basic neural network building blocks: `Neuron`, `Layer`, `MLP`

## Building

See [DEV_INSTALL.md](DEV_INSTALL.md) for the full setup (vcpkg + uv + CMake). Short version:

```sh
uv sync --dev
```

## Usage

```python
import zbtorch as zb

a = zb.Tensor([1.0, 2.0, 3.0], device="cpu")
b = zb.Tensor([4.0, 5.0, 6.0], device="cpu")

out = (a * b).tanh()
out.backward()

print(a.grad)
```

## Project layout

- `zbtorch/` — the installable package (C++/CUDA extension + Python wrapper)
  - `src/zbtorch_ext/` — C++/CUDA extension (tensor, ops, autograd, CPU/CUDA backends)
  - `src/zbtorch/` — Python package wrapping the extension
  - `src/zbtorch_py/` — earlier pure-Python prototype, kept for reference
- `scripts/` — demo/benchmark scripts (e.g. spiral classification on CPU/CUDA/pure-Python)
- `tests/` — pytest test suite
