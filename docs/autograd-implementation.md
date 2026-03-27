---
git_start: 7a8804b
git_end: fe7c0e6
date_range: 2026-03-20 ~ 2026-03-28
title: 从零实现自动求导系统：计算图与反向传播
description: 深入理解深度学习框架的核心——通过从零实现 Value 类和自动求导系统，掌握计算图构建、梯度计算和反向传播的原理
---

> **说明**: 本文档聚焦于自动求导系统的核心实现，省略了项目初始化配置、包管理、开发环境设置等基建内容。

## 第1章: 计算图构建基础

### 1.1 核心问题

深度学习框架（如 PyTorch、TensorFlow）的核心能力是**自动求导**（Autograd）。要实现自动求导，首先需要解决的问题是：**如何追踪复杂的计算过程？**

考虑一个简单的表达式：

$$
L = ((a \times b) + c) \times f
$$

当我们要计算 $L$ 对 $a$ 的梯度时，需要知道：
- $L$ 是如何一步步计算出来的
- $a$ 在计算过程中的哪个环节被使用
- 中间结果之间的依赖关系

### 1.2 设计思路

解决方案是构建**计算图**（Computational Graph），这是一种有向无环图（DAG），其中：
- **节点**表示数值（变量或中间结果）
- **边**表示数据的流动方向
- 每个节点记录其**来源**（来自哪些父节点）和**操作类型**

核心数据结构：
```python
class Value:
    def __init__(self, data, children=(), _op="", label=""):
        self.data = data          # 存储数值
        self._prev = set(children) # 记录父节点
        self._op = _op            # 记录操作类型 (+, *, tanh 等)
        self.label = label        # 节点标签（用于可视化）
```

**关键设计**：`_prev` 使用集合存储父节点，这构建了图的边。

### 1.3 具体实现

#### 运算符重载

通过重载 Python 运算符，自动构建计算图：

```python
def __add__(self, rhs: "Value | float"):
    rhs = rhs if isinstance(rhs, Value) else Value(rhs)
    out = Value(self.data + rhs.data, (self, rhs), "+")
    return out

def __mul__(self, rhs: "Value | float"):
    rhs = rhs if isinstance(rhs, Value) else Value(rhs)
    out = Value(self.data * rhs.data, (self, rhs), "*")
    return out
```

**执行流程**：
1. 当执行 `c = a + b` 时
2. 创建新节点 `c`，其 `data = a.data + b.data`
3. 设置 `c._prev = {a, b}`，记录依赖关系
4. 设置 `c._op = "+"`，记录操作类型

#### 示例：构建计算图

```python
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

e = a * b      # e._prev = {a, b}, e._op = "*"
e.label = "e"

d = e + c      # d._prev = {e, c}, d._op = "+"
d.label = "d"

f = Value(-2.0, label="f")
L = d * f      # L._prev = {d, f}, L._op = "*"
L.label = "L"
```

计算图结构：
```
a ----*---- e ----+---- d ----*---- L
       |          |          |
b -----+          c          f
```

### 1.4 可视化系统

为了直观理解计算图，实现了 GraphViz 可视化：

```python
def trace(root: Value):
    """遍历计算图，收集所有节点和边"""
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
    """生成 GraphViz 可视化"""
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root)

    for node in nodes:
        uid = str(id(node))
        # 节点显示: label | data
        dot.node(name=uid,
                 label=f"{node.label} | data {node.data:.4f}",
                 shape="record")
        # 操作节点
        if node._op:
            dot.node(name=uid + node._op, label=node._op)
            dot.edge(uid + node._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
```

**可视化效果**：每个节点显示标签和数值，操作节点以圆形表示，清晰展示数据流动路径。

---

## 第2章: 梯度追踪系统

### 2.1 核心问题

上一章构建了计算图，记录了**前向传播**过程。但要实现自动求导，还需要追踪**反向传播**过程，即计算梯度。

问题：**如何存储和展示每个节点的梯度？**

### 2.2 设计思路

梯度是标量值，表示最终输出对当前节点的偏导数。为每个 `Value` 对象添加 `grad` 属性：

```python
def __init__(self, data, children=(), _op="", label=""):
    self.data = data
    self._prev = set(children)
    self._op = _op
    self.label = label
    self.grad = 0.0  # 初始梯度为 0
```

**为什么初始梯度是 0？**
- 在反向传播开始前，我们不知道该节点对输出的影响
- 反向传播时，从输出节点开始，设置 `output.grad = 1.0`
- 然后逐步向输入节点传播梯度

### 2.3 可视化增强

在可视化时显示梯度：

```python
dot.node(name=uid,
         label=f"{node.label} | data {node.data:.4f} | grad {node.grad:.4f}",
         shape="record")
```

现在节点显示：`label | data | grad`

### 2.4 理解梯度含义

对于计算图 $L = ((a \times b) + c) \times f$：

- `a.grad` 表示 $\frac{\partial L}{\partial a}$
- `b.grad` 表示 $\frac{\partial L}{\partial b}$
- 依此类推

梯度告诉我们：**当某个节点的值发生微小变化时，最终输出会如何变化**。

---

## 第3章: 手动梯度验证

### 3.1 核心问题

在实现自动反向传播前，需要验证梯度计算的正确性。如何验证？

### 3.2 设计思路

使用**数值梯度**（Numerical Gradient）验证，基于导数的定义：

$$
\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

取足够小的 $h$（如 0.001），计算有限差分作为梯度的近似值。

### 3.3 具体实现

```python
def get_grad(i):
    """验证节点 i 的梯度"""
    delta = 0.001

    # 基准值
    sample = sample_data()  # 构建计算图
    L1 = sample.data

    # 微扰后的值
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    if i == "b":
        b.data += delta
    elif i == "a":
        a.data += delta
    # ... 重新构建计算图 ...
    L2 = L.data

    # 数值梯度
    print((L2 - L1) / delta)

# 验证 d 的梯度
get_grad("d")  # 输出: -2.000 (理论值: f = -2.0)
# 验证 f 的梯度
get_grad("f")  # 输出: 4.000 (理论值: d = 4.0)
```

### 3.4 链式法则推导

通过手动推导理解梯度传播：

#### 乘法操作

$$
L = d \times f
$$

求偏导：
$$
\frac{\partial L}{\partial d} = f, \quad \frac{\partial L}{\partial f} = d
$$

验证结果：`d.grad = -2.0`, `f.grad = 4.0`，与数值梯度一致。

#### 加法操作

$$
d = e + c
$$

求偏导：
$$
\frac{\partial d}{\partial e} = 1.0, \quad \frac{\partial d}{\partial c} = 1.0
$$

应用链式法则：
$$
\frac{\partial L}{\partial c} = \frac{\partial L}{\partial d} \times \frac{\partial d}{\partial c} = (-2.0) \times 1.0 = -2.0
$$

#### 乘法操作（反向传播）

$$
e = a \times b
$$

求偏导：
$$
\frac{\partial e}{\partial a} = b, \quad \frac{\partial e}{\partial b} = a
$$

应用链式法则：
$$
\frac{\partial L}{\partial a} = \frac{\partial L}{\partial e} \times \frac{\partial e}{\partial a} = (-2.0) \times (-3.0) = 6.0
$$

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial e} \times \frac{\partial e}{\partial b} = (-2.0) \times 2.0 = -4.0
$$

### 3.5 梯度优化示例

理解梯度后，可以进行简单的优化：

```python
# 手动设置梯度
a.grad = 6.0
b.grad = -4.0
c.grad = -2.0
f.grad = 4.0

# 沿梯度方向微调（学习率 = 0.01）
a.data += a.grad * 0.01
b.data += b.grad * 0.01
c.data += c.grad * 0.01
f.data += f.grad * 0.01

# 重新计算 L
L = ((a * b) + c) * f
print(L)  # 输出: -7.326 (原始值: -8.0)
```

**梯度下降原理**：
- 梯度指向函数值**增大**最快的方向
- 反方向移动，函数值会**减小**
- 重复此过程，可以最小化损失函数

---

## 第4章: 自动反向传播

### 4.1 核心问题

前几章通过手动计算和验证梯度，理解了反向传播的原理。但是：
- **手动计算繁琐**：每个节点都要手动推导链式法则
- **容易出错**：复杂的计算图容易遗漏路径
- **不可扩展**：无法处理动态计算图

如何实现**自动反向传播**？

### 4.2 设计思路

自动反向传播的关键是：
1. **局部反向传播**：每个操作节点知道如何计算自己的局部梯度
2. **拓扑排序**：确保按正确顺序反向传播
3. **梯度累积**：处理节点在多个路径被使用的情况

### 4.3 数学原理

#### 链式法则回顾

对于复合函数 $y = f(g(x))$：

$$
\frac{dy}{dx} = \frac{dy}{dg} \times \frac{dg}{dx}
$$

在计算图中，链式法则体现为：**每个节点接收来自下游的梯度，乘以局部梯度，传递给上游节点**。

#### 各操作的局部梯度

**加法操作** $o = a + b$：
$$
\frac{\partial o}{\partial a} = 1.0, \quad \frac{\partial o}{\partial b} = 1.0
$$

**乘法操作** $o = a \times b$：
$$
\frac{\partial o}{\partial a} = b, \quad \frac{\partial o}{\partial b} = a
$$

**指数操作** $o = e^a$：
$$
\frac{\partial o}{\partial a} = e^a = o
$$

**幂次操作** $o = a^n$：
$$
\frac{\partial o}{\partial a} = n \cdot a^{n-1}
$$

### 4.4 具体实现

#### 4.4.1 局部反向传播函数

在每个操作中嵌入 `_backward` 函数：

```python
def __add__(self, rhs: "Value | float"):
    rhs = rhs if isinstance(rhs, Value) else Value(rhs)
    out = Value(self.data + rhs.data, (self, rhs), "+")

    def _backward():
        # 链式法则: 局部梯度 × 下游梯度
        self.grad += 1.0 * out.grad
        rhs.grad += 1.0 * out.grad

    out._backward = _backward
    return out

def __mul__(self, rhs: "Value | float"):
    rhs = rhs if isinstance(rhs, Value) else Value(rhs)
    out = Value(self.data * rhs.data, (self, rhs), "*")

    def _backward():
        self.grad += rhs.data * out.grad
        rhs.grad += self.data * out.grad

    out._backward = _backward
    return out
```

**关键设计**：
- `_backward` 是一个**闭包**（Closure），捕获了 `self`, `rhs`, `out`
- 使用 `+=` 累积梯度，而非 `=`，处理节点被多次使用的情况
- 初始化时设置 `_backward = lambda: None`，处理叶子节点

#### 4.4.2 拓扑排序

反向传播必须按**从输出到输入**的顺序执行。使用拓扑排序确保正确顺序：

```python
def backward(self):
    # 构建拓扑排序
    topo = []
    visited = set()

    def build_topo(v: Value):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)  # 深度优先搜索
            topo.append(v)

    build_topo(self)

    # 设置输出节点梯度
    self.grad = 1.0

    # 反向传播
    for node in reversed(topo):
        node._backward()
```

**拓扑排序原理**：
- 深度优先遍历计算图
- 每个节点在其所有子节点之后被加入列表
- 反转后得到从输出到输入的顺序

示例：
```
计算图: a -> e -> d -> L
拓扑序: [a, b, e, c, d, f, L]
反转后: [L, f, d, c, e, b, a]  (正确的反向传播顺序)
```

### 4.5 梯度累积问题

> **为什么使用 `+=` 而不是 `=`？**

考虑这种情况：

```python
a = Value(3.0, label="a")
b = a + a  # b = 2a
b.backward()
```

计算图：
```
a ----+
      +---- (+) ---- b
a ----+
```

节点 `a` 被使用两次。正确的梯度：
$$
\frac{\partial b}{\partial a} = \frac{\partial (a+a)}{\partial a} = 1 + 1 = 2
$$

如果使用 `=`，第二次计算会覆盖第一次：
```python
self.grad = 1.0 * out.grad  # 错误: 覆盖
```

使用 `+=` 累积梯度：
```python
self.grad += 1.0 * out.grad  # 正确: 累积
```

**直觉理解**：一个节点对输出的贡献来自多个路径时，梯度应该**累加**。

---

## 第5章: 激活函数与运算扩展

### 5.1 核心问题

神经网络需要激活函数引入非线性。如何在 `Value` 类中实现激活函数？

### 5.2 tanh 激活函数

#### 数学定义

$$
\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}
$$

**特性**：
- 输出范围：$(-1, 1)$
- 在原点附近近似线性
- 两端饱和（梯度趋近 0）

#### 梯度推导

设 $o = \tanh(n)$，求 $\frac{\partial o}{\partial n}$：

$$
\frac{\partial \tanh(x)}{\partial x} = 1 - \tanh^2(x)
$$

证明：
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

使用商的导数公式：
$$
\frac{d}{dx}\left(\frac{u}{v}\right) = \frac{u'v - uv'}{v^2}
$$

计算得：
$$
\frac{\partial \tanh(x)}{\partial x} = 1 - \tanh^2(x) = 1 - o^2
$$

#### 实现

```python
def tanh(self):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (self,), _op="tanh")

    def _backward():
        self.grad += (1.0 - t**2) * out.grad

    out._backward = _backward
    return out
```

### 5.3 指数函数

#### 实现

```python
def exp(self):
    x = self.data
    out = Value(math.exp(x), (self,), "exp")

    def _backward():
        self.grad += out.data * out.grad  # d(e^x)/dx = e^x

    out._backward = _backward
    return out
```

#### tanh 的分解实现

使用基础运算符实现 `tanh`，验证计算图的正确性：

```python
def tanh_without_lib(self):
    e = (2 * self).exp()
    o = (e - 1) / (e + 1)
    return o
```

这会构建更复杂的计算图，但结果相同。

### 5.4 幂次运算

#### 数学推导

$$
o = a^n, \quad \frac{\partial o}{\partial a} = n \cdot a^{n-1}
$$

#### 实现

```python
def __pow__(self, rhs: "int | float"):
    assert isinstance(rhs, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**rhs, (self,), f"**{rhs}")

    def _backward():
        self.grad += rhs * (self.data ** (rhs - 1)) * out.grad

    out._backward = _backward
    return out
```

### 5.5 其他运算符

基于已有运算符实现：

```python
def __truediv__(self, rhs):  # self / rhs
    return self * rhs**-1

def __neg__(self):  # -self
    return self * -1

def __sub__(self, other):  # self - other
    return self + (-other)
```

### 5.6 反向操作符

支持 Python 运算符的对称性：

```python
def __radd__(self, lhs: "Value | float"):
    return self.__add__(lhs)

def __rmul__(self, lhs: "Value | float"):
    return self.__mul__(lhs)
```

**作用**：使得 `2 * a` 和 `a * 2` 都能正常工作。

### 5.7 神经元示例

综合运用所有功能，构建单个神经元：

```python
# 输入
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

# 权重
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# 偏置
b = Value(6.881337, label="b")

# 前向传播
x1w1 = x1 * w1
x2w2 = x2 * w2
n = x1w1 + x2w2 + b
o = n.tanh()  # 输出

# 反向传播
o.backward()
```

计算图自动构建，梯度自动计算，完整实现了神经元的正向和反向传播。

---

## 第6章: 代码模块化

### 6.1 核心问题

随着功能增加，`grad.ipynb` 文件变得庞大。如何组织代码结构？

### 6.2 设计思路

将核心类提取到独立文件：
- `value.py`: Value 类
- `MLP.ipynb`: 神经网络实现
- `grad.ipynb`: 教学演示和可视化

### 6.3 具体实现

创建 `value.py`：

```python
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

    # ... 所有方法 ...

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
```

在 notebook 中使用：

```python
from value import Value
```

### 6.4 Neuron 类

开始构建神经网络抽象：

```python
import random

class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
```

这为后续构建多层感知器（MLP）奠定基础。

---

## 总结

### 核心成果

从零实现了完整的自动求导系统：

1. **计算图构建**：通过运算符重载自动追踪计算过程
2. **梯度追踪**：每个节点存储梯度信息
3. **自动反向传播**：基于拓扑排序和链式法则自动计算梯度
4. **激活函数**：实现 `tanh`、`exp` 等非线性变换
5. **可视化系统**：GraphViz 直观展示计算图结构

### 设计亮点

- **闭包设计**：每个操作嵌入 `_backward` 函数，实现局部反向传播
- **拓扑排序**：确保反向传播的正确顺序
- **梯度累积**：使用 `+=` 处理节点被多次使用的情况
- **运算符重载**：提供直观的数学表达式语法

### 扩展方向

- 添加更多激活函数（ReLU、Sigmoid）
- 实现 Layer 和 MLP 类
- 添加优化器（SGD、Adam）
- 实现批量计算
- GPU 加速

### 关键洞察

自动求导的核心思想是：
1. **前向传播**：构建计算图，记录数据流动
2. **反向传播**：按拓扑序反向传递梯度
3. **链式法则**：每个节点只负责局部梯度计算

这个简单的系统展示了 PyTorch、TensorFlow 等框架的核心原理。理解这些基础概念，是掌握深度学习框架的关键。
