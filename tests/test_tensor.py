import numpy as np
import pytest
import torch
from zbtorch import Tensor, cuda


DEVICES = ["cpu"] + (["cuda"] if cuda.is_available() else [])


def check(zb: Tensor, pt: torch.Tensor, tol: float = 1e-5):
    """Assert forward values and gradients match between zbtorch and PyTorch."""
    assert np.allclose(zb.data, pt.detach().numpy().ravel(), atol=tol), \
        f"data mismatch:\n  zbtorch: {zb.data}\n  pytorch: {pt.detach().numpy().ravel()}"
    if pt.grad is not None:
        assert np.allclose(np.asarray(zb.grad), pt.grad.numpy().ravel(), atol=tol), \
            f"grad mismatch:\n  zbtorch: {zb.grad}\n  pytorch: {pt.grad.numpy().ravel()}"


def make(data, device="cpu"):
    """Create matching zbtorch and PyTorch tensors (float32)."""
    arr = np.array(data, dtype=np.float32)
    zb = Tensor(arr, device=device)
    pt = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
    return zb, pt


# --- Construction ---

def test_construct_scalar():
    t = Tensor(3.14)
    assert np.isclose(t.data[0], 3.14, atol=1e-5)


def test_construct_numpy():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = Tensor(arr)
    assert np.allclose(t.data, arr)


def test_construct_vec_shape():
    # vec+shape constructor requires numpy for now (pybind11/stl.h not included,
    # so python list -> vector<size_t> conversion is unavailable)
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = Tensor(arr)
    assert np.allclose(t.data, [1.0, 2.0, 3.0, 4.0])


# --- Properties ---

def test_data_length_1d():
    t = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert len(t.data) == 3


def test_data_length_2d():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    t = Tensor(arr)
    assert len(t.data) == 6


@pytest.mark.parametrize("device", DEVICES)
def test_device_property(device):
    t = Tensor(np.array([1.0], dtype=np.float32), device=device)
    assert t.device == device


def test_data_setter():
    t = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    t.data = np.array([5.0, 6.0], dtype=np.float32)
    assert np.allclose(t.data, [5.0, 6.0])


# --- zero_grad ---

@pytest.mark.parametrize("device", DEVICES)
def test_zero_grad(device):
    za, _ = make([1.0, 2.0, 3.0], device)
    zo = za + za
    zo.backward()
    assert not np.all(np.asarray(za.grad) == 0), "grad should be non-zero before zero_grad"
    za.zero_grad()
    assert np.all(np.asarray(za.grad) == 0)


# --- matmul ---

@pytest.mark.parametrize("device", DEVICES)
def test_matmul(device):
    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b_data = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)

    za = Tensor(a_data, device=device)
    zb = Tensor(b_data, device=device)
    zo = za @ zb
    zo.backward()

    pa = torch.tensor(a_data, requires_grad=True)
    pb = torch.tensor(b_data, requires_grad=True)
    po = torch.matmul(pa, pb)
    po.sum().backward()

    check(za, pa, tol=1e-4)
    check(zb, pb, tol=1e-4)
    check(zo, po, tol=1e-4)


# --- Scalar (right-hand) operators ---

@pytest.mark.parametrize("device", DEVICES)
def test_radd(device):
    za, pa = make([1.0], device)
    zo = 2.0 + za;  zo.backward()
    po = 2.0 + pa;  po.sum().backward()
    check(za, pa); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_rmul(device):
    za, pa = make([3.0, 4.0], device)
    zo = 3.0 * za;  zo.backward()
    po = 3.0 * pa;  po.sum().backward()
    check(za, pa); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_rsub(device):
    za, pa = make([1.0], device)
    zo = 5.0 - za;  zo.backward()
    po = 5.0 - pa;  po.sum().backward()
    check(za, pa); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_rtruediv(device):
    za, pa = make([2.0], device)
    zo = 1.0 / za;  zo.backward()
    po = 1.0 / pa;  po.sum().backward()
    check(za, pa); check(zo, po)


# --- Device transfer ---

@pytest.mark.parametrize("device", DEVICES)
def test_to_same_device(device):
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = Tensor(arr, device=device)
    t2 = t.to(device)
    assert t2.device == device
    assert np.allclose(t2.data, arr)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
def test_to_cuda():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = Tensor(arr, device="cpu")
    t_cuda = t.to("cuda")
    assert t_cuda.device == "cuda"
    assert np.allclose(t_cuda.data, arr)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            import inspect
            sig = inspect.signature(fn)
            if "device" in sig.parameters:
                for dev in DEVICES:
                    fn(dev)
                    print(f"  {name}[{dev}] passed")
            else:
                fn()
                print(f"  {name} passed")
