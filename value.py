from typing import List
import math


class Value:

    def __init__(self, data, children=(), _op="", label=""):
        self.data = data
        self._prev = set["Value"](children)
        self._op = _op
        self.label = label
        self.grad = 0.0

        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, rhs: "Value | float"):
        rhs = rhs if isinstance(rhs, Value) else Value(rhs)
        out = Value(self.data + rhs.data, (self, rhs), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            rhs.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, lhs: "Value| float"):
        return self.__add__(lhs)

    def __mul__(self, rhs: "Value| float"):
        rhs = rhs if isinstance(rhs, Value) else Value(rhs)

        out = Value(self.data * rhs.data, (self, rhs), "*")

        def _backward():

            self.grad += rhs.data * out.grad
            rhs.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, lhs: "Value| float"):
        return self.__mul__(lhs)

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), _op="tanh")

        def _backward():

            self.grad += (1.0 - t**2.0) * out.grad

        out._backward = _backward
        return out

    def tanh_without_lib(self):
        e = (2 * self).exp()
        o = (e - 1) / (e + 1)
        return o

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, rhs: "int| float"):
        assert isinstance(rhs, (int, float)), "only supporting int/float powers for now"

        out = Value(self.data**rhs, (self,), f"**{rhs}")

        def _backward():
            self.grad += rhs * (self.data ** (rhs - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, rhs):
        return self * rhs**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def backward(self):
        topo: List[Value] = []
        visited = set["Value"]()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
