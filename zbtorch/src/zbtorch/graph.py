import numpy as np
from pathlib import Path
import graphviz
from zbtorch_ext import build_topo


# def _trace(root) -> tuple[set, set]:
#     nodes: set = set()
#     edges: set = set()
#
#     def build(v):
#         if v not in nodes:
#             nodes.add(v)
#             for child in v._children:
#                 edges.add((v, child))
#                 build(child)
#
#     build(root)
#     return nodes, edges


def draw_graph(root, filename: str | Path | None = "graph") -> graphviz.Digraph:
    nodes = build_topo(root)
    dot = graphviz.Digraph(
        format="svg",
        graph_attr={"rankdir": "LR"},
        node_attr={"fontsize": "11"},
    )

    for n in nodes:
        uid = str(id(n))
        label = "{ %s | shape %s | grad %s }" % (
            n._label,
            tuple(n.shape),
            np.array2string(np.asarray(n.grad), precision=4),
        )
        dot.node(uid, label=label, shape="record", style="filled", fillcolor="#e8f4fc")
        if n._op:
            op_id = uid + n._op
            dot.node(op_id, label=n._op, style="filled", fillcolor="#ffe8cc")
            dot.edge(op_id, uid)

    for n1, n2 in edges:
        # n1 is the output Tensor; n2 is one of its inputs — wire input → op → output
        op_id = str(id(n1)) + n1._op
        dot.edge(str(id(n2)), op_id)

    if filename is not None:
        dot.render(str(filename), cleanup=True)

    return dot
