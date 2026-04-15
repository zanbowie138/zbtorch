import numpy as np
from zbtorch import Tensor, MLP, draw_graph

a = Tensor(5.0, label="a")
b = Tensor(6.0, label="b")
c = a * b; c._label = "c"
d = c + Tensor(5.0); d._label = "d"


d.backward()
draw_graph(d)
print("graph created.")