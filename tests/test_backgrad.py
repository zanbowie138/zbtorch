import numpy as np
import torch
from zbtorch import Tensor


def check(zb: Tensor, pt: torch.Tensor, tol: float = 1e-6):
    """Assert forward values and gradients match between zbtorch and PyTorch."""
    assert np.allclose(zb.data, pt.detach().numpy(), atol=tol), \
        f"data mismatch:\n  zbtorch: {zb.data}\n  pytorch: {pt.detach().numpy()}"
    if pt.grad is not None:
        assert np.allclose(np.asarray(zb.grad), pt.grad.numpy(), atol=tol), \
            f"grad mismatch:\n  zbtorch: {zb.grad}\n  pytorch: {pt.grad.numpy()}"


def make(data):
    """Create matching zbtorch and PyTorch tensors."""
    arr = np.array(data, dtype=np.float64)
    zb = Tensor(arr)
    pt = torch.tensor(arr, dtype=torch.float64, requires_grad=True)
    return zb, pt


def test_add():
    za, pa = make([1.0, 2.0])
    zb, pb = make([3.0, 4.0])
    zo = za + zb;  zo.backward()
    po = pa + pb;  po.sum().backward()
    check(za, pa); check(zb, pb); check(zo, po)


def test_mul():
    za, pa = make([2.0, 3.0])
    zb, pb = make([4.0, 5.0])
    zo = za * zb;  zo.backward()
    po = pa * pb;  po.sum().backward()
    check(za, pa); check(zb, pb); check(zo, po)


def test_pow():
    za, pa = make([2.0, 3.0])
    zo = za ** 3;  zo.backward()
    po = pa ** 3;  po.sum().backward()
    check(za, pa); check(zo, po)


def test_sub():
    za, pa = make([5.0, 6.0])
    zb, pb = make([1.0, 2.0])
    zo = za - zb;  zo.backward()
    po = pa - pb;  po.sum().backward()
    check(za, pa); check(zb, pb); check(zo, po)


def test_div():
    za, pa = make([6.0, 8.0])
    zb, pb = make([2.0, 4.0])
    zo = za / zb;  zo.backward()
    po = pa / pb;  po.sum().backward()
    check(za, pa); check(zb, pb); check(zo, po)


def test_neg():
    za, pa = make([3.0, -1.0])
    zo = -za;  zo.backward()
    po = -pa;  po.sum().backward()
    check(za, pa); check(zo, po)


def test_exp():
    za, pa = make([1.0, 2.0])
    zo = za.exp();  zo.backward()
    po = pa.exp();  po.sum().backward()
    check(za, pa); check(zo, po)


def test_log():
    za, pa = make([1.0, 2.0])
    zo = za.log();  zo.backward()
    po = pa.log();  po.sum().backward()
    check(za, pa); check(zo, po)


def test_relu():
    za, pa = make([-1.0, 0.0, 2.0])
    zo = za.relu();  zo.backward()
    po = pa.relu();  po.sum().backward()
    check(za, pa); check(zo, po)


def test_tanh():
    za, pa = make([0.0, 1.0, -1.0])
    zo = za.tanh();  zo.backward()
    po = pa.tanh();  po.sum().backward()
    check(za, pa); check(zo, po)


def test_sigmoid():
    za, pa = make([0.0, 1.0, -2.0])
    zo = za.sigmoid();  zo.backward()
    po = pa.sigmoid();  po.sum().backward()
    check(za, pa); check(zo, po)


def test_compound():
    """Multi-op expression: ((a * b) + a.exp()) / (b ** 2)"""
    za, pa = make([1.0, 2.0])
    zb, pb = make([3.0, 4.0])
    zo = (za * zb + za.exp()) / (zb ** 2);  zo.backward()
    po = (pa * pb + pa.exp()) / (pb ** 2);  po.sum().backward()
    check(za, pa); check(zb, pb); check(zo, po)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} passed")
