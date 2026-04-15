import numpy as np
from zbtorch._C import Tensor


class Neuron:
    def __init__(self, n_inputs: int):
        self.w = [Tensor(np.random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Tensor(0.0)

    def __call__(self, x: list[Tensor]) -> Tensor:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self) -> list[Tensor]:
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = [0.0] * len(p.data)


class Layer:
    def __init__(self, n_inputs: int, n_outputs: int):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x: list[Tensor]) -> list[Tensor]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Tensor]:
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = [0.0] * len(p.data)


class MLP:
    def __init__(self, n_inputs: int, layer_sizes: list[int]):
        sizes = [n_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]

    def __call__(self, x: list[float]) -> list[Tensor]:
        out = [Tensor(xi) for xi in x]
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self) -> list[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = [0.0] * len(p.data)
