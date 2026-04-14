import math
from pathlib import Path
import graphviz

# Based heavily on Andrej Karpathy's tutorial: https://www.youtube.com/watch?v=VMj-3S1tku0

def _trace(root: "Value") -> tuple[set, set]:
    nodes: set = set()
    edges: set = set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((v, child))
                build(child)

    build(root)
    return nodes, edges


def draw_graph(root: "Value", filename: str | Path | None = "graph") -> graphviz.Digraph:
    nodes, edges = _trace(root)
    dot = graphviz.Digraph(
        format="svg",
        graph_attr={"rankdir": "LR"},
        node_attr={"fontsize": "11"},
    )

    for n in nodes:
        uid = str(id(n))
        label = "{ %s | data %.4f | grad %.4f }" % (n._label, n.data, n.grad)
        dot.node(uid, label=label, shape="record", style="filled", fillcolor="#e8f4fc")
        if n._op:
            op_id = uid + n._op
            dot.node(op_id, label=n._op, style="filled", fillcolor="#ffe8cc")
            dot.edge(op_id, uid)

    for n1, n2 in edges:
        # n1 is the output value; n2 is one of its inputs — wire input → op → output
        op_id = str(id(n1)) + n1._op
        dot.edge(str(id(n2)), op_id)

    if filename is not None:
        dot.render(str(filename), cleanup=True)

    return dot


class Value:
    def __init__(self, data, _children=(), _op='', _label=''):
        self.data = float(data)
        self._children = set(_children)
        self._op = _op
        self._label = _label
        self._backward = lambda: None
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op='{self._op}', label='{self._label}')"

    # ------------------------------------------------------------------
    # Core ops
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "__pow__ only supports scalar exponents"
        out = Value(self.data ** other, (self,), f'**{other}')

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
    # Right-hand variants (e.g. 2 + Value, 3 * Value)
    # ------------------------------------------------------------------

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __rtruediv__(self, other):
        return Value(other) * self ** -1

    # ------------------------------------------------------------------
    # Activations / transcendentals
    # ------------------------------------------------------------------

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad  # d/dx e^x = e^x

        out._backward = _backward
        return out

    def log(self):
        assert self.data > 0, "log is only defined for positive values"
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backprop
    # ------------------------------------------------------------------

    def backward(self):
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        print(topo)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


a = Value(2.0, _label='a')
b = Value(3.0, _label='b')
c = a * b; c._label = 'c'
d = c * 5; d._label = 'd'

d.backward()
draw_graph(d)
