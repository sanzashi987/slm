---
git_start: 7a8804b
git_end: 4e1c5d9
date_range: 2026-03-20 ~ 2026-03-31
title: 从零实现自动微分与多层感知机：计算图、反向传播与梯度下降
description: 从最简单的 Value 类出发，逐步引入计算图追踪、数值验证、链式法则、自动反向传播，最终构建完整的 MLP 训练循环，彻底理解深度学习框架的核心机制
---

> **说明**: 本文档聚焦于自动微分系统和多层感知机的核心算法实现，省略了项目初始化配置、GraphViz 环境安装、checkpoint 文件管理等辅助内容。

## 1: 计算图——追踪计算历史

### 1.1 核心问题

要实现自动微分，首先要解决一个基础问题：**如何记住一个值是怎么算出来的？**

假设有表达式 $L = (a \cdot b + c) \cdot f$，当我们要计算 $\partial L / \partial a$ 时，需要知道 $L$ 的完整计算路径。普通的浮点数不携带任何计算历史，需要一个能"记住来龙去脉"的新数据类型。

### 1.2 Value 类的初始设计

最初版本的 `Value` 只支持加法和乘法，但已经建立了计算图的核心思路：

```python
class Value:
    def __init__(self, data, children=(), _op="", label=""):
        self.data = data
        self._prev = set(children)   # 产生此值的输入节点
        self._op = _op               # 产生此值的运算符
        self.label = label
        self.grad = 0.0              # 梯度（暂时为 0）

    def __add__(self, rhs):
        return Value(self.data + rhs.data, (self, rhs), "+")

    def __mul__(self, rhs):
        return Value(self.data * rhs.data, (self, rhs), "*")
```

每次运算都创建一个新的 `Value` 节点，并通过 `_prev` 指向参与运算的子节点。多次运算串联后，就自动构建出一棵有向无环图（DAG）——这就是计算图。

### 1.3 用 GraphViz 可视化计算图

```python
from graphviz import Digraph

def trace(root: Value):
    """深度优先遍历，收集所有节点和边"""
    nodes, edges = set(), set()
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
        # 显示节点的 label、data 和 grad
        dot.node(name=uid,
                 label=f"{node.label} | data {node.data:.4f} | grad {node.grad:.4f}",
                 shape="record")
        if node._op:
            # 为运算符单独创建一个节点
            dot.node(name=uid + node._op, label=node._op)
            dot.edge(uid + node._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot
```

可视化效果：每个值节点显示为矩形，内含 data 和 grad；运算符节点（`+`、`*`）位于两者之间，箭头方向表示数据流向（从输入到输出）。

---

## 2: 梯度的含义——从数值验证到链式法则

### 2.1 核心问题

有了计算图，下一步是理解**梯度**在其中的含义，并验证我们的推导是否正确。

给定表达式 $L = d \cdot f$，其中 $d = e + c$，$e = a \cdot b$：

$$L = (a \cdot b + c) \cdot f$$

$\partial L / \partial a$ 等于多少？如何不靠直觉、用数学方法验证？

### 2.2 数值微分验证

**数值微分**（Numerical Differentiation）用极限定义近似计算梯度：

$$\frac{\partial L}{\partial x} \approx \frac{L(x + h) - L(x)}{h}, \quad h \to 0$$

```python
def get_grad(var_name, delta=0.001):
    """固定其他变量，对指定变量施加 delta，观察 L 的变化"""
    a = Value(2.0); b = Value(-3.0)
    c = Value(10.0); f = Value(-2.0)
    L1 = ((a * b + c) * f).data   # 基准值

    # 对目标变量加 delta
    if var_name == 'a': a.data += delta
    if var_name == 'b': b.data += delta
    if var_name == 'f': f.data += delta

    L2 = ((a * b + c) * f).data   # 变化后的值
    print((L2 - L1) / delta)      # 近似梯度

get_grad('d')  # → -2.0  (L = d * f, dL/dd = f = -2.0)
get_grad('f')  # → 4.0   (L = d * f, dL/df = d = 4.0)
get_grad('a')  # → 6.0
get_grad('b')  # → -4.0
```

数值验证的结果和手动推导完全吻合，这建立了后续实现自动反向传播的信心。

### 2.3 链式法则：梯度如何"传递"

设 $L = d \cdot f$，$d = e + c$，$e = a \cdot b$。从输出到输入逐步推导：

**第一层（乘法节点 $L = d \cdot f$）**:

$$\frac{\partial L}{\partial d} = f = -2.0, \quad \frac{\partial L}{\partial f} = d = 4.0$$

**第二层（加法节点 $d = e + c$）**:

加法的局部梯度是 1（两个输入对输出的偏导都等于 1），用链式法则：

$$\frac{\partial L}{\partial c} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial c} = (-2.0) \cdot 1.0 = -2.0$$
$$\frac{\partial L}{\partial e} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial e} = (-2.0) \cdot 1.0 = -2.0$$

**第三层（乘法节点 $e = a \cdot b$）**:

乘法的局部梯度：$\partial e / \partial a = b$，$\partial e / \partial b = a$：

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial e} \cdot b = (-2.0) \cdot (-3.0) = 6.0$$
$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial e} \cdot a = (-2.0) \cdot 2.0 = -4.0$$

> **链式法则的直觉**
> 链式法则 $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x}$ 的含义是：
> - 上游梯度（$\partial L / \partial z$）表示"L 对这个节点的输出有多敏感"
> - 局部梯度（$\partial z / \partial x$）表示"这个节点的输出对其输入有多敏感"
> - 两者相乘，得到"L 对这个节点的输入有多敏感"
>
> 反向传播就是在计算图上从输出到输入逐层应用链式法则。

---

## 3: 自动反向传播——让计算图"自己求导"

### 3.1 核心问题

手动推导梯度对小表达式可行，但对有数百万参数的神经网络完全不可能。我们需要让计算图**自动**完成反向传播，每个节点知道如何将梯度传递给它的输入。

### 3.2 _backward 闭包：局部梯度的封装

关键设计是为每个运算节点附加一个 `_backward` 函数，它知道如何把"输出的梯度"转化为"输入的梯度"：

```python
def __add__(self, rhs):
    out = Value(self.data + rhs.data, (self, rhs), "+")

    def _backward():
        # 加法：局部梯度为 1，上游梯度直接传递
        # 用 += 而非 =，支持同一节点被多次使用（梯度累积）
        self.grad += 1.0 * out.grad
        rhs.grad  += 1.0 * out.grad

    out._backward = _backward
    return out

def __mul__(self, rhs):
    out = Value(self.data * rhs.data, (self, rhs), "*")

    def _backward():
        # 乘法：局部梯度是对方的值
        self.grad += rhs.data * out.grad
        rhs.grad  += self.data * out.grad

    out._backward = _backward
    return out
```

`_backward` 是一个**闭包**，它捕获了运算时的 `self`、`rhs`、`out` 引用。运算结束后，这些值的关系被永久封存在函数里，随时可以调用。

> **为什么用 `+=` 而不是 `=`？**
> 当同一个 `Value` 节点被多次使用时（如 $a$ 出现在 $a \cdot b$ 和 $a \cdot c$ 中），它会在计算图中被多条路径的梯度共同影响。用 `+=` 将来自不同路径的梯度累积起来，等价于多元链式法则中的求和项：
> $$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial e_1} \cdot \frac{\partial e_1}{\partial a} + \frac{\partial L}{\partial e_2} \cdot \frac{\partial e_2}{\partial a}$$

### 3.3 拓扑排序：确保梯度的传递顺序

反向传播必须按照**从输出到输入**的顺序调用 `_backward`，确保每个节点在调用 `_backward` 时，其输出节点的梯度已经完整累积。

用拓扑排序实现这一顺序：

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v: Value):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)   # 先递归到叶节点
            topo.append(v)          # 后处理（后序遍历）

    build_topo(self)
    # topo 中，叶节点在前，输出节点在后
    # 反转后：从输出节点到叶节点
    self.grad = 1.0           # dL/dL = 1
    for node in reversed(topo):
        node._backward()      # 每个节点将梯度传给其 children
```

> **为什么是后序遍历（post-order）？**
> 后序遍历保证：当处理节点 $v$ 时，所有以 $v$ 为子节点的父节点都已经处理完毕，即 $v.grad$ 已经从所有上游路径完整累积。`reversed(topo)` 将后序列表倒转，就得到了从输出到输入的正确顺序。

### 3.4 激活函数：tanh 及其导数

单个加法和乘法不足以让神经网络学习非线性关系，需要引入激活函数。tanh 是常用选择，其定义和导数：

$$\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$$

$$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$

```python
def tanh(self):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (self,), _op="tanh")

    def _backward():
        # 局部梯度：d(tanh(x))/dx = 1 - tanh(x)²
        self.grad += (1.0 - t ** 2) * out.grad

    out._backward = _backward
    return out
```

### 3.5 Cross-Entropy：分类任务的损失函数

tanh 输出的是 $(-1, 1)$ 的连续值，适合回归任务（如本文第 5 章的 MLP 预测 ±1）。但在多分类任务（如 GPT 预测下一个 token）中，网络的最终输出是一个未归一化的得分向量（logits），需要一个配套的损失函数——Cross-Entropy。

**从 Softmax 到概率分布**

给定 $C$ 个类别的 logits $z_1, z_2, \ldots, z_C$，Softmax 将其转为概率分布：

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

每个 $p_i \in (0, 1)$ 且 $\sum p_i = 1$，可解释为模型对"输入属于第 $i$ 类"的置信度。

**Cross-Entropy 的定义**

设真实类别为 $y$（one-hot），模型预测概率为 $p$：

$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{C} y_i \log p_i = -\log p_y$$

由于 one-hot 向量只有目标类别 $y$ 处为 1，其余为 0，损失退化为**只看目标类别的对数概率的负值**。$p_y$ 越接近 1（预测越自信且正确），$-\log p_y$ 越接近 0；$p_y$ 越小（预测错误或不自信），损失越大。

**为什么不用 MSE？**

对分类任务使用 MSE（$(p_y - 1)^2$）有一个严重问题：当 $p_y$ 接近 0（完全预测错误）时，MSE 的梯度 $2(p_y - 1)$ 约等于 $-2$，仍然是有限值；而 $-\log p_y$ 在 $p_y \to 0$ 时趋向无穷大，给出强烈的梯度信号，推动模型快速纠正错误。

**梯度推导**

将 Softmax + Cross-Entropy 合并推导（避免数值不稳定），损失对 logits 的梯度极其简洁：

$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_i} = p_i - y_i$$

**推导过程**：

$$\mathcal{L} = -\log p_y = -z_y + \log\!\sum_j e^{z_j}$$

对 $z_i$ 求偏导：

$$\frac{\partial \mathcal{L}}{\partial z_i} = -\mathbf{1}[i=y] + \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i - y_i$$

> **这个梯度的直觉**
> 对目标类别 $i = y$：梯度为 $p_y - 1$，当 $p_y$ 很小时梯度接近 $-1$，推动 $z_y$ 增大。
> 对非目标类别 $i \neq y$：梯度为 $p_i$，当 $p_i$ 很大时（错误地给非目标类高分）梯度接近 $+1$，推动 $z_i$ 减小。
> 整体效果：把概率质量从错误类别转移到正确类别。

在 PyTorch 中，`F.cross_entropy(logits, target)` 内部已将 Softmax 和 Cross-Entropy 合并计算（LogSoftmax + NLLLoss），避免了先显式计算 Softmax 再取 log 带来的数值精度问题。

### 3.6 扩展运算符——复用已有的反向传播

完整的 `Value` 类还支持幂运算、除法、减法、取负。关键洞察是这些运算可以通过已有运算组合而来，**无需单独实现反向传播**：

```python
def __pow__(self, rhs: int | float):
    out = Value(self.data ** rhs, (self,), f"**{rhs}")

    def _backward():
        # 幂函数导数：d(x^n)/dx = n * x^(n-1)
        self.grad += rhs * (self.data ** (rhs - 1)) * out.grad

    out._backward = _backward
    return out

def __truediv__(self, rhs):
    return self * rhs ** -1     # a/b = a * b^(-1)，复用乘法和幂运算

def __neg__(self):
    return self * -1            # -a = a * (-1)

def __sub__(self, other):
    return self + (-other)      # a - b = a + (-b)

def __radd__(self, lhs):        # 支持 sum([v1, v2, v3]) 等内置函数
    return self.__add__(lhs)
```

`__radd__` 和 `__rmul__` 处理右操作数为 `Value` 时的情况（如 `3 * v`），让 Python 内置的 `sum()` 函数也能对 `Value` 列表求和。

---

## 4: 单个神经元——从数学到代码

### 4.1 生物学类比与数学抽象

一个神经元的计算可以用以下公式描述：

$$o = \tanh\!\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

其中 $x_i$ 是输入，$w_i$ 是权重，$b$ 是偏置，$\tanh$ 是激活函数。

```python
class Neuron:
    def __init__(self, nin: int):
        # 权重和偏置用 Value 包装，以便参与自动微分
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: list[float]):
        # sum(wi*xi, start=b) 利用 __radd__ 支持浮点数 x_i 与 Value wi 相乘
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]   # 返回所有可训练参数
```

`__call__` 让 `Neuron` 实例可以像函数一样被调用：`n([2.0, 3.0])` 等价于 `n.__call__([2.0, 3.0])`。

---

## 5: 多层感知机——从神经元到网络

### 5.1 Layer 与 MLP

将多个神经元组合为层，多个层堆叠为 MLP：

```python
class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        sz = [nin] + nouts
        # 相邻两层的大小配对：(nin, nouts[0]), (nouts[0], nouts[1]), ...
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)     # 逐层前向传播
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

`MLP(3, [4, 4, 1])` 创建一个 3 个输入、两个隐藏层（各 4 个神经元）、1 个输出的网络，共有 $3\times4 + 4 + 4\times4 + 4 + 4\times1 + 1 = 41$ 个参数。

### 5.2 损失函数

均方误差（MSE）是监督学习最常用的损失函数：

$$\mathcal{L} = \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

```python
inputs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
targets = [1.0, -1.0, -1.0, 1.0]

def compute_loss(mlp):
    youts = [mlp(x) for x in inputs]
    losses = [(yout - ytarget) ** 2 for yout, ytarget in zip(youts, targets)]
    return sum(losses)   # 利用 __radd__ 对 Value 列表求和
```

`sum(losses)` 中，`losses` 是 `Value` 列表，`sum` 内部会调用 `__radd__`（整数 `0 + Value(...)`），最终返回一个 `Value` 节点，该节点携带完整的计算图，可以直接调用 `.backward()`。

---

## 6: 训练循环——梯度下降的完整实现

### 6.1 梯度下降

有了 `loss.backward()` 计算出所有参数的梯度，梯度下降更新参数：

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta}$$

其中 $\eta$ 是学习率（learning rate）。对 `loss` 越大的方向，梯度越大，参数更新的步伐也越大，逐渐把 loss 推向最小值。

```python
def gradient_descent(mlp, eta: float):
    for p in mlp.parameters():
        p.data += -1 * p.grad * eta  # 沿梯度反方向移动
```

### 6.2 梯度清零：一个关键细节

在 `Value._backward()` 的实现中，梯度用 `+=` 累积（支持节点被多次引用）。这意味着**每次训练迭代前，必须手动将所有参数的梯度归零**，否则新一轮的梯度会和上一轮的梯度叠加，导致参数更新错误：

```python
def make_training(mlp, inputs, targets):
    def training(iterates: int, eta: float):
        for k in range(iterates):
            # 前向传播
            youts = [mlp(x) for x in inputs]
            losses = [(yout - ytarget) ** 2 for yout, ytarget in zip(youts, targets)]
            loss = sum(losses)

            # !! 关键：每次反向传播前归零梯度 !!
            for p in mlp.parameters():
                p.grad = 0.0

            # 反向传播
            loss.backward()

            # 梯度下降
            for p in mlp.parameters():
                p.data += -1 * p.grad * eta

            print(k, loss.data)

    return training
```

> **为什么不在 `backward()` 内部自动清零？**
> 将清零留给调用方是有意的设计。在某些高级训练场景（如梯度累积，用多个小 batch 模拟大 batch），需要在多次 `backward()` 后才执行一次参数更新，此时梯度必须跨 batch 累积而不是清零。显式清零让调用方对训练过程有完整控制权。

---

## 总结

本文从零构建了一个完整的自动微分系统和多层感知机训练流程：

- **计算图**：每次运算创建新的 `Value` 节点，`_prev` 链接输入，形成有向无环图
- **数值验证**：用 $(f(x+h) - f(x))/h$ 验证梯度推导的正确性，建立对链式法则的直觉
- **自动反向传播**：`_backward` 闭包封装局部梯度，拓扑排序保证梯度从输出到输入的正确传递顺序，`+=` 支持节点多次引用时的梯度累积
- **激活函数**：tanh 引入非线性，其导数 $1 - \tanh^2(x)$ 直接融入 `_backward`；除法、减法等通过组合已有运算实现，无需额外反向传播代码
- **MLP 架构**：Neuron → Layer → MLP 三层抽象，`parameters()` 收集所有可训练参数
- **训练循环**：前向传播计算 MSE 损失，梯度归零（防止跨 batch 累积），`backward()` 自动求导，梯度下降更新参数
