---
git_start: 868abcb
git_end: 4e1c5d9
date_range: 2026-03-28 ~ 2026-03-31
title: 多层感知机自动微分系统实现
description: 从单个神经元到完整训练流程的演进过程
---

> **说明**: 本文档聚焦于多层感知机(MLP)的核心实现和自动微分系统，省略了项目配置、checkpoint管理等辅助内容。

## 1: 神经元基础实现

### 1.1 核心问题

如何从零开始实现一个可自动求导的神经元？神经元是神经网络的基本计算单元，需要能够：
- 存储权重和偏置参数
- 执行前向传播计算
- 支持反向传播计算梯度

### 1.2 设计思路

神经元的设计基于生物学神经元的数学抽象：
- **权重(Weights)**: 每个输入对应一个权重，表示该输入的重要性
- **偏置(Bias)**: 调节神经元的激活阈值
- **激活函数**: 使用tanh引入非线性

关键设计决策是使用自定义的`Value`类包装所有数值，使计算过程可追踪，为自动微分奠定基础。

### 1.3 数学原理

**线性组合**:
$$
z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w} \cdot \mathbf{x} + b
$$

其中：
- $w_i$ 是第 $i$ 个输入的权重
- $x_i$ 是第 $i$ 个输入值
- $b$ 是偏置项

**tanh激活函数**:
$$
\text{tanh}(z) = \frac{e^{2z} - 1}{e^{2z} + 1}
$$

**tanh的导数**:
$$
\frac{d}{dz}\text{tanh}(z) = 1 - \text{tanh}^2(z)
$$

> **为什么选择tanh？** tanh函数将输出压缩到[-1, 1]区间，相比sigmoid的[0, 1]区间，其输出以0为中心，有助于加速梯度下降收敛。同时，tanh在两端的梯度更平缓，有助于缓解梯度消失问题。

### 1.4 具体实现

```python
class Neuron:
    def __init__(self, nin: int):
        # 随机初始化权重和偏置，范围[-1, 1]
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: List[float]):
        # 线性组合: w·x + b
        act = sum([wi * xi for (wi, xi) in zip(self.w, x)], self.b)
        # 应用激活函数
        out = act.tanh()
        return out

    def parameters(self):
        # 返回所有可训练参数
        return self.w + [self.b]
```

**使用示例**:
```python
# 创建一个接收2个输入的神经元
n = Neuron(2)
# 前向传播
output = n([2.0, 3.0])
```

---

## 2: 多层网络架构

### 2.1 核心问题

单个神经元的能力有限，如何组织多个神经元形成强大的多层神经网络？需要解决：
- 如何管理一层中的多个神经元？
- 如何连接多个层形成深度网络？
- 如何实现数据在不同层间的流动？

### 2.2 设计思路

采用分层架构设计：
1. **Layer**: 将多个神经元组织成层，实现并行计算
2. **MLP**: 将多个层按顺序连接，形成完整网络

这种设计遵循了"组合优于继承"的原则，通过简单的组合关系构建复杂的网络结构。

**网络拓扑示意**:
```
输入层 (3个特征)
    ↓
隐藏层1 (4个神经元)
    ↓
隐藏层2 (4个神经元)
    ↓
输出层 (1个神经元)
```

### 2.3 具体实现

**Layer类实现**:
```python
class Layer:
    def __init__(self, nin: int, nout: int):
        # 创建nout个神经元，每个接收nin个输入
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # 并行执行层内所有神经元
        outs = [n(x) for n in self.neurons]
        # 如果只有一个输出，直接返回Value；否则返回列表
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # 收集层内所有神经元的参数
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
```

**MLP类实现**:
```python
class MLP:
    def __init__(self, nin: int, nouts: List[int]):
        # 构建网络结构：[nin, nout1, nout2, ..., noutN]
        sz = [nin] + nouts
        # 创建层，每层的输入是上一层的输出
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        # 数据依次流过每一层
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # 收集所有层的参数
        return [p for layer in self.layers for p in layer.parameters()]
```

**使用示例**:
```python
# 创建网络: 3个输入，两个隐藏层(各4个神经元)，1个输出
mlp = MLP(3, [4, 4, 1])
# 前向传播
output = mlp([2.0, 3.0, -1.0])
```

---

## 3: 参数管理与损失函数

### 3.1 核心问题

有了网络结构，如何评估网络的性能并调整参数？需要：
- 统一管理所有可训练参数
- 定义量化评估指标（损失函数）
- 实现梯度计算和参数更新

### 3.2 设计思路

**参数管理**:
- 递归收集策略：Neuron → Layer → MLP
- 统一的`parameters()`接口，便于上层调用

**损失函数选择**:
- 采用均方误差(MSE)衡量预测与目标的差距
- MSE对异常值敏感，适合回归任务

### 3.3 数学原理

**均方误差(MSE)**:
$$
L = \frac{1}{n}\sum_{i=1}^{n}(y_{\text{pred}}^{(i)} - y_{\text{target}}^{(i)})^2
$$

在实现中使用求和形式（省略1/n系数，不影响优化方向）:
$$
L = \sum_{i=1}^{n}(y_{\text{pred}}^{(i)} - y_{\text{target}}^{(i)})^2
$$

**梯度下降更新规则**:
$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \frac{\partial L}{\partial \theta}
$$

其中：
- $\theta$ 表示参数（权重或偏置）
- $\eta$ 是学习率，控制更新步长
- $\frac{\partial L}{\partial \theta}$ 是损失对参数的梯度

> **学习率的选择**: 学习率太大可能导致震荡不收敛，太小则收敛速度慢。通常从0.1或0.01开始尝试。

### 3.4 具体实现

**损失计算示例**:
```python
# 定义训练数据
inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
targets = [1.0, -1.0, -1.0, 1.0]

# 创建MLP
mlp = MLP(3, [4, 4, 1])

# 前向传播
youts = [mlp(x) for x in inputs]

# 计算损失
losses = [(yout - ytarget) ** 2 for (yout, ytarget) in zip(youts, targets)]
loss = sum(losses)

# 反向传播
loss.backward()

# 梯度下降更新
for p in mlp.parameters():
    p.data += -1 * p.grad * 0.01  # 学习率 = 0.01
```

---

## 4: 完整训练流程

### 4.1 核心问题

如何将训练过程封装成可复用的组件？需要解决：
- 避免重复代码
- 实现标准的训练循环
- 正确处理梯度累积问题

### 4.2 设计思路

**梯度累积问题**:
在自动微分中，`Value`类使用`+=`累积梯度，以支持参数在计算图中多次出现。这要求在每个训练迭代开始前必须重置梯度。

**训练循环抽象**:
使用闭包(Closure)模式，创建包含网络状态和训练数据的训练器，返回可配置的训练函数。

### 4.3 数学原理

**梯度累积的必要性**:
考虑一个参数在计算图中出现多次的情况：
```
y = w × x₁ + w × x₂
```
根据链式法则：
$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot x_1 + \frac{\partial L}{\partial y} \cdot x_2
$$

因此梯度必须是累加的。

**梯度重置**:
每次迭代前将所有参数的梯度设为0，避免上一轮的梯度影响当前计算。

### 4.4 具体实现

**可复用的训练组件**:
```python
def make_training(mlp: MLP, inputs: List[List[float]], targets: List[float]):
    """
    创建训练器

    Args:
        mlp: 神经网络
        inputs: 输入数据
        targets: 目标输出

    Returns:
        training函数，接受迭代次数和学习率
    """
    def training(iterates: int, eta: float):
        for k in range(iterates):
            # 前向传播
            youts = [mlp(x) for x in inputs]
            losses = [(yout - ytarget) ** 2 for (yout, ytarget) in zip(youts, targets)]
            loss = sum(losses)

            # !!! 关键：重置梯度
            for p in mlp.parameters():
                p.grad = 0.0

            # 反向传播
            loss.backward()

            # 参数更新
            for p in mlp.parameters():
                p.data += -1 * p.grad * eta

            # 打印训练进度
            print(k, loss.data)

    return training
```

**使用示例**:
```python
# 创建网络和数据
mlp = MLP(3, [4, 4, 1])
inputs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
targets = [1.0, -1.0, -1.0, 1.0]

# 创建训练器
training = make_training(mlp, inputs, targets)

# 执行训练：20次迭代，学习率0.1
training(20, 0.1)
```

**训练输出示例**:
```
0 7.888351481428312
1 7.850965288821508
2 7.7843302477498835
...
10 0.49860893031692466
11 0.010185290334303768
...
19 0.00813120363709422
```

可以看到损失从7.89逐渐下降到0.008，训练成功收敛。

---