---
git_start: 73f26b4
git_end: 52f6da7
date_range: 2026-04-02 ~ 2026-04-15
title: 从零实现 GPT：从 Bigram 到 Transformer 的完整演进
description: 沿着 Karpathy 路线，从最简单的 Bigram 语言模型出发，逐步引入注意力机制、Multi-Head、残差连接、LayerNorm，最终实现可加载 GPT-2 权重的完整 Transformer，并加入 Flash Attention、梯度累积、学习率调度等工程优化
---

> **说明**: 本文档聚焦于语言模型的核心实现和训练工程，省略了数据集下载、Jupyter 执行顺序调整等辅助内容。代码演进从 `gpt-dev.ipynb` 原型开始，最终迁移到 `gpt-2/train.py` 生产代码。

## 1: 语言建模基础——Bigram 模型

### 1.1 核心问题

语言模型的本质任务是：**给定已见的 token 序列，预测下一个 token 的概率分布**。

最简单的起点是 Bigram 模型——它只用当前 token 预测下一个 token，完全忽略更长的上下文。虽然极其简单，但它提供了一个完整的训练-生成管线，后续所有复杂性都建立在这个骨架上。

### 1.2 字符级 Tokenizer

在使用 GPT-2 的 BPE tokenizer 之前，先用字符级 tokenizer 建立直觉：

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])
```

字符集合大小即词表大小（Tiny Shakespeare 约 65 个字符）。

### 1.3 批次采样

训练数据的组织方式直接影响模型能学到什么。一个关键设计是：**一个 chunk 同时提供 T 个训练样本**——位置 t 处的前 t 个 token 预测第 t+1 个 token：

```python
def get_batch(split, batch_size=4, block_size=8):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])      # (B, T)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # (B, T)
    return x, y
```

`x[b, :t]` 是上下文，`y[b, t]` 是目标。每次 `get_batch` 返回形状 `(B, T)` 的张量对，等价于 `B * T` 个独立的预测任务。

### 1.4 Bigram 模型实现

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # Embedding 表直接作为 logits 查表：token i → 预测下一个 token 的得分向量
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        else:
            loss = None
        return logits, loss
```

> **为什么 Embedding 维度等于 vocab_size？**
> Bigram 模型中，Embedding 表的每一行直接就是"看到 token i 后，下一个 token 是 j 的得分"，形状 `(vocab_size, vocab_size)`。查表操作 `embedding(idx)` 等价于 one-hot 向量乘以权重矩阵，输出就是 logits。这意味着整个模型只有一个 `vocab_size × vocab_size` 的参数矩阵——极其简单，但完全忽略了位置和上下文。

### 1.5 自回归生成

```python
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :]          # 取最后一步的 logits: (B, vocab_size)
        probs = F.softmax(logits, dim=-1)  # 转为概率分布
        idx_next = torch.multinomial(probs, num_samples=1)  # 采样: (B, 1)
        idx = torch.cat([idx, idx_next], dim=1)             # (B, T+1)
    return idx
```

每次循环生成一个 token，并将其拼接到序列末尾作为下一次的输入——这就是"自回归"（autoregressive）的含义。

---

## 2: 注意力机制——从均值聚合到 Scaled Dot-Product

### 2.1 核心问题

Bigram 的致命缺陷是：每个 token 只能看到自己，无法利用历史上下文。但如果允许每个位置"回顾"之前所有位置，怎么做？

最朴素的想法：对历史 token 的表示取平均。

### 2.2 三种等价的均值聚合

**版本 1：嵌套循环（直觉清晰，效率极差）**

```python
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]         # (t, C)
        xbow[b, t] = xprev.mean(0) # 对时间维度取均值
```

**版本 2：下三角矩阵乘法（高效）**

```python
weight = torch.tril(torch.ones(T, T))
weight /= weight.sum(dim=1, keepdim=True)  # 每行归一化
xbow2 = weight @ x  # (T,T) @ (B,T,C) -> (B,T,C)，自动广播 batch 维
```

下三角矩阵保证位置 t 只聚合 0..t 的信息（causal）。

**版本 3：Softmax 掩码（为注意力机制铺路）**

```python
tril = torch.tril(torch.ones(T, T))
weight = torch.zeros((T, T))
weight = weight.masked_fill(tril == 0, float("-inf"))
weight = F.softmax(weight, dim=-1)  # -inf 位置变为精确的 0
xbow3 = weight @ x
```

> **为什么用 `-inf` 而不是 `0`？**
> `softmax(0) = 1/T ≠ 0`，不能表示"完全忽略"。而 `softmax(-inf) = e^{-∞} / Z = 0`，精确地把未来 token 的权重置零。这个 masked softmax 的思路直接沿用到 Self-Attention 的因果掩码中。

### 2.3 Self-Attention：让权重"数据相关"

均值聚合的缺陷：每个 token 对历史所有位置等权重关注，忽略了"哪些历史 token 更相关"这一关键信息。

Self-Attention 的解法：用可学习的线性变换生成 Query（我需要什么）、Key（我能提供什么）、Value（我实际携带的信息），通过 Q·K^T 计算相关性作为注意力权重：

```python
def self_attention_single_head():
    B, T, C = 4, 8, 32
    x = torch.randn(B, T, C)
    head_size = 16

    query = nn.Linear(C, head_size, bias=False)  # 我需要什么信息
    key   = nn.Linear(C, head_size, bias=False)  # 我能提供什么信息
    value = nn.Linear(C, head_size, bias=False)  # 我实际携带什么

    q = query(x)  # (B, T, head_size)
    k = key(x)    # (B, T, head_size)
    v = value(x)  # (B, T, head_size)

    # 注意力权重：q 和 k 的点积，结果越大表示越"相关"
    wei = q @ k.transpose(-2, -1)  # (B, T, T)
    wei = wei * head_size ** -0.5  # 缩放：防止 softmax 饱和
    wei = wei.masked_fill(tril == 0, float("-inf"))  # 因果掩码
    wei = F.softmax(wei, dim=-1)   # 归一化为概率

    out = wei @ v  # (B, T, head_size)
```

### 2.4 为什么要除以 $\sqrt{d_k}$？

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

设 $Q, K$ 的元素 $\sim \mathcal{N}(0, 1)$，则点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$，标准差为 $\sqrt{d_k}$。

当 $d_k$ 很大时（如 64），点积的数量级也随之增大，softmax 会急剧饱和：

$$
\text{softmax}([1, 2, 3]) \approx [0.09, 0.24, 0.67] \quad \text{（分散）}
$$
$$
\text{softmax}([10, 20, 30]) \approx [0, 0, 1] \quad \text{（退化为 argmax）}
$$

饱和的 softmax 导致梯度几乎为 0，训练困难。除以 $\sqrt{d_k}$ 把方差归一化回 1，保持梯度流动。

---

## 3: 多头注意力与 Transformer Block

### 3.1 单头 → 多头

单个注意力头只能关注一种"相关性模式"（如句法依赖、语义相似度等）。多头注意力并行运行 $h$ 个头，每个头独立学习不同的关注模式，最后拼接输出：

```python
class Head(nn.Module):
    def __init__(self, n_embed, head_size, block_size):
        super().__init__()
        self.key   = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, block_size, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size) for _ in range(n_head)])
        self.proj  = nn.Linear(n_embed, n_embed)  # 输出投影

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)
```

`n_head` 个头各自输出 `(B, T, head_size)`，拼接后得 `(B, T, n_head * head_size) = (B, T, n_embed)`。

### 3.2 FeedForward：逐位置的 MLP

注意力机制负责 token 间的"通信"（gathering information），完成后每个位置的表示需要独立地"思考"（computation）。这由 FeedForward 完成：

```python
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)
```

中间层扩展到 `4 * n_embed` 是 Transformer 原论文的设计，给模型足够的"计算空间"。

### 3.3 残差连接与 Layer Normalization

深层网络的两个问题：梯度消失（信号无法传回早期层）和训练不稳定（各层输出的数值分布差异大）。

**残差连接**在每个子层（注意力 / FFN）外包一层 `x = x + sublayer(x)`，梯度可以直接从输出流回输入，不经过子层：

```python
class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        head_size = n_embed // n_head
        self.sa   = MultiHeadAttention(n_embed, block_size, n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln_1 = nn.LayerNorm(n_embed)
        self.ln_2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln_1(x))    # 注意力 + 残差
        x = x + self.ffwd(self.ln_2(x))  # FFN + 残差
        return x
```

> **Pre-norm vs Post-norm**
> 原始论文是 Post-norm（`LayerNorm(x + sublayer(x))`），但现代实现（包括 GPT-2）改为 Pre-norm（`x + sublayer(LayerNorm(x))`），因为训练更稳定——Layer Norm 在子层输入时就规范化，防止梯度在深层累积爆炸。

**Layer Normalization** 对每个样本的特征维度独立做均值 0、方差 1 的归一化，不依赖 batch 大小：

$$
\text{LN}(x_i) = \gamma \cdot \frac{x_i - \mu}{\sigma + \epsilon} + \beta, \quad \mu = \frac{1}{d}\sum_j x_j, \quad \sigma = \sqrt{\frac{1}{d}\sum_j (x_j - \mu)^2}
$$

---

## 4: GPT 完整架构

### 4.1 从 notebook 迁移到 train.py

在 gpt-dev.ipynb 中验证所有组件后，代码被重构为 `gpt-2/train.py`（后拆分为 `classes.py` + `train.py`），架构和 GPT-2 原始实现对齐，以便后续加载预训练权重。

### 4.2 GPTConfig 与模型结构

```python
@dataclass
class GPTConfig:
    block_size: int = 1024   # 最大上下文长度
    vocab_size: int = 50257  # 50000 BPE + 256 bytes token + 1 <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768        # 嵌入维度

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),  # token embedding
            "wpe": nn.Embedding(config.block_size, config.n_embd),  # position embedding
            "h":   nn.ModuleList([Layer(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),  # 最终 LayerNorm
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

前向传播：token embedding + position embedding → N 个 Transformer Block → 最终 LayerNorm → lm_head 投影到词表：

```python
def forward(self, idx, target=None):
    B, T = idx.shape
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    x = self.transformer.wte(idx) + self.transformer.wpe(pos)  # (B, T, n_embd)
    for block in self.transformer.h:
        x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)  # (B, T, vocab_size)

    loss = None
    if target is not None:
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))
    return logits, loss
```

### 4.3 权重共享（Weight Tying）

```python
self.transformer.wte.weight = self.lm_head.weight
```

`wte`（token embedding）和 `lm_head` 的形状都是 `(vocab_size, n_embd)`，但职责方向相反：

- `wte`：token index → 语义向量（编码）
- `lm_head`：语义向量 → logits（解码），本质是 `x @ wte.weight.T`

两者学的是同一件事：**"token i 的语义向量是什么"**。共享权重不仅减少约 3800 万参数（vocab_size=50257, n_embd=768），还强制了编码与解码的一致性。

### 4.4 高效的多头注意力实现

相比 notebook 版本（为每个 head 独立分配 Linear），生产版本将 Q、K、V 的线性变换合并为一个矩阵：

```python
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # Q+K+V 合并
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)                       # (B, T, 3*n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)   # 各 (B, T, n_embd)
        head_size = C // self.n_head
        # 展开 head 维度：(B, T, n_embd) → (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        # Flash Attention（见第5章）
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 合并 heads
        return self.c_proj(y)
```

关键操作：`.view(B, T, n_head, head_size).transpose(1, 2)` 将 n_embd 维度分拆为 n_head 个 head，每个 head 独立计算注意力，最后用 `.view` 还原。这等价于 notebook 中 `torch.cat([h(x) for h in self.heads], dim=-1)`，但计算效率高得多。

---

## 5: 训练工程优化

### 5.1 DataLoaderLite：高效数据加载

```python
class DataLoaderLite:
    def __init__(self, B: int, T: int):
        self.B, self.T = B, T
        with open("input.txt", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(text))
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0  # 一个 epoch 结束，从头开始
        return x, y
```

将完整文本预先 tokenize 为 flat tensor，每次 `next_batch` 用滑动窗口切片，避免重复编码开销。

### 5.2 Flash Attention：从 $O(T^2)$ 到 $O(T)$ 的显存访问

传统注意力的瓶颈不是计算量，而是显存带宽：

```
传统流程：
1. att = q @ k.T  → 写入显存 [T, T] 矩阵
2. att = softmax(att)  → 从显存读回，计算，再写回
3. y = att @ v  → 再从显存读回
```

当 T=1024 时，`[T, T]` 矩阵有 100 万个元素，反复读写成为瓶颈。

Flash Attention 的解法：将 `[T, T]` 矩阵**分块**，在 GPU 片上 SRAM（速度比显存快 ~10x）里完成计算，利用 online softmax 技巧在不看完整 `[T, T]` 矩阵的情况下完成分块 softmax，只把最终结果 `y` 写回显存。数学结果完全等价，显存读写次数从 $O(T^2)$ 降到 $O(T)$。

PyTorch 2.0 已内置：

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 5.3 权重初始化：残差累积的数学控制

残差连接带来一个隐患：$N$ 层 Block，每层做 2 次残差（Attention + FFN），方差累积：

$$
\text{Var}(x_N) = \text{Var}(x_0) + 2N \cdot \sigma^2
$$

层数越深，激活值的方差越大，训练越不稳定。解决方法是让每个残差分支的输出层（`c_proj`）初始化更小的权重，抵消累积效应：

$$
\sigma_{\text{c\_proj}} = \frac{0.02}{\sqrt{2N}}
$$

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        std = 0.02
        if hasattr(module, "RES_CONNECT"):
            std *= (2 * self.config.n_layer) ** -0.5  # 残差分支专用缩放
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

`c_proj` 通过 `self.c_proj.RES_CONNECT = 1` 标记为残差分支最后一层。12 层 Block 下：`std = 0.02 / sqrt(24) ≈ 0.004`，每次加入残差流的量只有普通层的 1/5。

### 5.4 AdamW 与参数分组

AdamW 对不同参数使用不同的 weight decay 策略：

```python
def configure_optimizer(self, weight_decay, learning_rate, device):
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]  # 2D 权重矩阵
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]   # 1D 偏置/归一化

    optim_groups = [
        {"params": decay_params,   "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    return optimizer
```

> **为什么偏置和 LayerNorm 参数不做 weight decay？**
> Weight decay 的物理含义是惩罚权重的 L2 范数，防止过拟合。但 LayerNorm 的 $\gamma, \beta$ 和线性层的 bias 是 1D 参数，没有"权重过大"的过拟合风险；对它们做 decay 反而会干扰归一化行为。GPT-3 论文的实践：只对 2D 权重矩阵（Embedding、Linear 的 weight）做 decay。

AdamW 更新规则（对比 Adam）：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{（梯度方向）}
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{（梯度幅度）}
$$
$$
\theta_t = \theta_{t-1} - \alpha \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \alpha \cdot \lambda \cdot \theta_{t-1}
$$

AdamW 把 weight decay 从梯度里分离出来（最后一项），避免 Adam 中 adaptive learning rate 削弱 weight decay 效果的问题。

### 5.5 学习率调度：Warmup + Cosine Decay

```python
max_lr, min_lr = 3e-4, 3e-5  # min_lr = max_lr * 0.1
warmup_steps, max_steps = 10, 50

def get_lr(it: int):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps   # 线性 warmup
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # cosine: 1 → 0
    return min_lr + coeff * (max_lr - min_lr)
```

三个阶段的设计动机：

| 阶段 | 学习率 | 原因 |
|------|--------|------|
| Warmup（0→max_lr） | 线性增加 | 训练初期权重随机，梯度方向不可信，大学习率会把权重推到奇怪位置 |
| 中期（max_lr） | 保持峰值 | 梯度方向已稳定，大学习率加速收敛 |
| 后期（max_lr→min_lr） | Cosine 下降 | 接近最优时大学习率会在最优点附近震荡，慢慢降低让模型稳定落点 |

Cosine 函数的优点：下降曲线平滑，避免线性 decay 末期学习率骤降带来的不稳定。

### 5.6 梯度累积：用串行模拟大 Batch

显存限制了单次能处理的 batch 大小，梯度累积允许用多个小 batch 模拟大 batch：

```python
total_batch_size = 524288  # 2**19 ≈ 0.5M tokens，接近 GPT-3 的设置
B, T = 16, 1024
grad_accum_steps = total_batch_size // (B * T)  # = 32

optimizer.zero_grad()
loss_accum = 0.0

for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss = model(x, y)

    # cross_entropy 默认取平均，多步累积需要手动修正：
    # (L1+L2)/2 + (L3+L4)/2 ≠ (L1+L2+L3+L4)/4
    loss = loss / grad_accum_steps   # 修正：等价于对所有 micro batch 统一取平均
    loss_accum += loss.detach()
    loss.backward()   # 梯度在 param.grad 里累加，不清零

norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

> **梯度范数裁剪的作用**
> `clip_grad_norm_` 计算所有参数梯度的 L2 范数 $\|g\|_2 = \sqrt{\sum g_i^2}$，若超过 `max_norm=1.0`，则等比缩小所有梯度使范数恰好等于 1.0。这防止某个异常 batch 导致梯度爆炸，是大规模训练的标准做法（GPT-3 也使用 `max_norm=1.0`）。

### 5.7 Vocab Size 对齐与 BFloat16

```python
model = GPT(GPTConfig(vocab_size=50304))  # 50257 → 50304（128 的倍数）
```

GPU 矩阵运算（CUDA kernel）在维度是 2 的幂次或 128 的倍数时效率最高。50257 是质数，改为 50304 = 393×128 让 lm_head 矩阵乘法命中高效内核。多出的 47 个 token（50257~50303）不存在于词表，训练时从不出现这些 index，对应权重被训练成极低得分，实践中模型几乎不会采样到。

混合精度训练：

```python
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    _, loss = model(x, y)

torch.set_float32_matmul_precision("high")  # TF32：10 位尾数，约 3x 加速
```

BFloat16 保留 FP32 的指数范围（8 位），只截短尾数（23 位 → 7 位），避免数值溢出。对比 FP16（更窄指数范围），BFloat16 训练更稳定，是 A100 等现代 GPU 的推荐格式。

---

## 总结

本文记录了语言模型从 Bigram 到 GPT 的完整演进：

- **Bigram** 提供了完整的训练-生成骨架，但无法利用上下文
- **Self-Attention** 通过 Q/K/V 机制让每个 token 按相关性聚合历史信息，Scaled Dot-Product 和因果掩码是核心
- **Multi-Head + FFN + 残差 + LayerNorm** 组成 Transformer Block，残差连接解决梯度消失，Pre-norm 让深层训练稳定
- **GPT 架构** 将 token embedding 和 position embedding 相加，N 个 Block 堆叠，权重共享节省参数并强制语义一致性
- **训练工程** 包含五个相互配合的优化：Flash Attention（IO 效率）、权重初始化（残差方差控制）、AdamW 参数分组（weight decay 精细化）、Warmup + Cosine 调度（稳定收敛）、梯度累积（突破显存限制模拟大 batch）
