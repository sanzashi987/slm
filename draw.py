import os
import math
import numpy as np
import matplotlib.pyplot as plt
from value import Value

graphviz_path = r"c:\\Program Files\\Graphviz\bin"

# 将路径添加到系统的 PATH 环境变量中
os.environ["PATH"] += os.pathsep + graphviz_path

from graphviz import Digraph


def trace(root: Value):
    nodes, edges = set[Value](), set()

    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root: Value):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root)
    for node in nodes:
        uid = str(id(node))
        dot.node(
            name=uid,
            label=f"{node.label} | data {node.data:.4f} | grad {node.grad:.4f}",
            shape="record",
        )
        if node._op:
            dot.node(name=uid + node._op, label=node._op)
            dot.edge(uid + node._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot
