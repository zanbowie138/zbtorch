import numpy as np
import pytest
import torch
from zbtorch import Tensor, cuda


DEVICES = ["cpu"] + (["cuda"] if cuda.is_available() else [])


def check(zb: Tensor, pt: torch.Tensor, tol: float = 1e-6):
    """Assert forward values and gradients match between zbtorch and PyTorch."""
    assert np.allclose(zb.data, pt.detach().numpy(), atol=tol), \
        f"data mismatch:\n  zbtorch: {zb.data}\n  pytorch: {pt.detach().numpy()}"
    if pt.grad is not None:
        assert np.allclose(np.asarray(zb.grad), pt.grad.numpy(), atol=tol), \
            f"grad mismatch:\n  zbtorch: {zb.grad}\n  pytorch: {pt.grad.numpy()}"


def make(data, device="cpu"):
    """Create matching zbtorch and PyTorch tensors."""
    arr = np.array(data, dtype=np.float64)
    zb = Tensor(arr, device=device)
    pt = torch.tensor(arr, dtype=torch.float64, requires_grad=True)
    return zb, pt


@pytest.mark.parametrize("device", DEVICES)
def test_add(device):
    za, pa = make([1.0, 2.0], device)
    zb, pb = make([3.0, 4.0], device)
    zo = za + zb;  zo.backward()
    po = pa + pb;  po.sum().backward()
    check(za, pa); check(zb, pb); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_mul(device):
    za, pa = make([2.0, 3.0], device)
    zb, pb = make([4.0, 5.0], device)
    zo = za * zb;  zo.backward()
    po = pa * pb;  po.sum().backward()
    check(za, pa); check(zb, pb); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_pow(device):
    za, pa = make([2.0, 3.0], device)
    zo = za ** 3;  zo.backward()
    po = pa ** 3;  po.sum().backward()
    check(za, pa); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_sub(device):
    za, pa = make([5.0, 6.0], device)
    zb, pb = make([1.0, 2.0], device)
    zo = za - zb;  zo.backward()
    po = pa - pb;  po.sum().backward()
    check(za, pa); check(zb, pb); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_div(device):
    za, pa = make([6.0, 8.0], device)
    zb, pb = make([2.0, 4.0], device)
    zo = za / zb;  zo.backward()
    po = pa / pb;  po.sum().backward()
    check(za, pa); check(zb, pb); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_neg(device):
    za, pa = make([3.0, -1.0], device)
    zo = -za;  zo.backward()
    po = -pa;  po.sum().backward()
    check(za, pa); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_exp(device):
    za, pa = make([1.0, 2.0], device)
    zo = za.exp();  zo.backward()
    po = pa.exp();  po.sum().backward()
    check(za, pa); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_log(device):
    za, pa = make([1.0, 2.0], device)
    zo = za.log();  zo.backward()
    po = pa.log();  po.sum().backward()
    check(za, pa); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_relu(device):
    za, pa = make([-1.0, 0.0, 2.0], device)
    zo = za.relu();  zo.backward()
    po = pa.relu();  po.sum().backward()
    check(za, pa); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_tanh(device):
    za, pa = make([0.0, 1.0, -1.0], device)
    zo = za.tanh();  zo.backward()
    po = pa.tanh();  po.sum().backward()
    check(za, pa); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_sigmoid(device):
    za, pa = make([0.0, 1.0, -2.0], device)
    zo = za.sigmoid();  zo.backward()
    po = pa.sigmoid();  po.sum().backward()
    check(za, pa); check(zo, po)


@pytest.mark.parametrize("device", DEVICES)
def test_compound(device):
    """Multi-op expression: ((a * b) + a.exp()) / (b ** 2)"""
    za, pa = make([1.0, 2.0], device)
    zb, pb = make([3.0, 4.0], device)
    zo = (za * zb + za.exp()) / (zb ** 2);  zo.backward()
    po = (pa * pb + pa.exp()) / (pb ** 2);  po.sum().backward()
    check(za, pa); check(zb, pb); check(zo, po)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            for dev in DEVICES:
                fn(dev)
                print(f"  {name}[{dev}] passed")
