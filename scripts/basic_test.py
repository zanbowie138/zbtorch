import numpy as np
from zbtorch import Tensor, MLP, draw_graph

a = Tensor(np.array([5.0]), _label="a")
b = Tensor(np.array([6.0]), _label="b")
c = a * b; c._label = "c"
d = c + 5; d._label = "d"


d.backward()
draw_graph(d)
