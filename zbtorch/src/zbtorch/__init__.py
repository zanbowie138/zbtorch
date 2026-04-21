from zbtorch._C import Tensor, Neuron, Layer, MLP, device, _cuda_is_available
from .graph import draw_graph


class _CudaNamespace:
    @staticmethod
    def is_available() -> bool:
        return _cuda_is_available()


cuda = _CudaNamespace()
