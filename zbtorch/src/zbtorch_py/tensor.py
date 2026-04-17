import numpy as np
from pathlib import Path
import graphviz

# Based heavily on Andrej Karpathy's tutorial: https://www.youtube.com/watch?v=VMj-3S1tku0


class Tensor:
    def __init__(self, data, _children=(), _op='', _label='', device="cpu"):
        self.data = np.asarray(data, dtype=float)
        self._children = set(_children)
        self._op = _op
        self._label = _label
        self._backward = lambda: None
        self.grad = 0.0
        self.device = device

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op='{self._op}', label='{self._label}')"

    # ------------------------------------------------------------------
    # Core ops
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "__pow__ only supports scalar exponents"
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    # ------------------------------------------------------------------
    # Right-hand variants (e.g. 2 + Tensor, 3 * Tensor)
    # ------------------------------------------------------------------

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Tensor(other) + (-self)

    def __rtruediv__(self, other):
        return Tensor(other) * self ** -1

    # ------------------------------------------------------------------
    # Activations / transcendentals
    # ------------------------------------------------------------------

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad  # d/dx e^x = e^x

        out._backward = _backward
        return out

    def log(self):
        assert np.all(self.data > 0), "log is only defined for positive Tensors"
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backprop
    # ------------------------------------------------------------------

    def backward(self):
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
