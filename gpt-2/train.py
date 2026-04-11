from dataclasses import dataclass
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch import Tensor


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # 50000 BPE + 256 bytes token + 1 eof token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class SelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd & config.n_head == 0
        # q,k,v linear matrix all in one
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.RES_CONNECT = 1

        self.n_head = config.n_head
        # q,k,v will be a four-dimensional tensor, so the low triangle matrix
        # also needs to be reshaped.
        self.register_buffer(
            "bias",  # key 'bias' as gpt-2 use this name
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_embd = config.n_embd

    def forward(self, x: Tensor):
        B, T, C = x.shape

        qkv: Tensor = self.c_attn(x)  # (B,T, 3*n_head)
        qkv_spilt: tuple[Tensor, Tensor, Tensor] = qkv.split(self.n_embd, dim=2)
        q, k, v = qkv_spilt
        head_size = C // self.n_head
        # introduce the `n_head` which resulting a tensor in (B, n_head, T, head_size)
        # just like the torch.cat in `MultiHead` getting (B , T, head_size) * n_head results
        # and concat them manually
        q = q.view(B, T, self.n_head, head_size).transpose(
            1, 2
        )  # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, head_size).transpose(
            1, 2
        )  # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, head_size).transpose(
            1, 2
        )  # (B, n_head, T, head_size)
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, n_head, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        # (B, n_head, T, T) x (B, n_head, T, head_size) = (B, n_head, T, head_size)
        y = att @ v
        y = y.transpose(1, 2)  # (B, T, n_head , head_size)
        y = y.contiguous().view(B, T, C)  # (B, T, n_embd)   n_embd = n_head * head_size
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")  # difference to  Relu
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.RES_CONNECT = 1

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Layer(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# 证明: 残差链接的方差通过加法累积
def deviation_grows():
    x = torch.zeros(768)
    n = 100
    for i in range(n):
        x += torch.randn(768) * n**-0.5
    print("deviation controlled:", x.std())
    x = torch.zeros(768)
    for i in range(n):
        x += torch.randn(768)
    print("deviation blows:", x.std())


# deviation_grows()


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Layer(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),  # additional layer norm
            }
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享（Weight Tying）：lm_head 和 wte 共享同一个 tensor
        #
        # wte      的形状：[vocab_size, n_embd]
        # lm_head  的形状：[vocab_size, n_embd]
        #
        # 两者形状相同，但职责方向相反：
        #   wte:     token_index → 向量  （把 token 编码成语义向量）
        #   lm_head: 向量 → logits       （把输出向量映射回词表得分）
        #
        # lm_head 的本质是矩阵点积：
        #   logits = x @ wte.weight.T
        #   即：用输出向量 x 和词表里每个 token 的 embedding 做相似度计算
        #   x 和 token 47 的 embedding 越相似 → token 47 的 logits 越高
        #
        # 因此两者学的是同一件事：
        #   "token 47 的语义向量是什么"
        #   这个向量既是 token 47 的输入表示（wte），
        #   也是输出时判断"当前向量最像哪个 token"的基准（lm_head）
        #
        # 共享的不是输出结果，而是对每个 token 语义的理解
        #
        # 额外好处：节省参数
        #   vocab_size=50257, n_embd=768 → 省去约 3800 万个参数
        self.transformer.wte.weight = self.lm_head.weight

        # the apply method from Module, to iterate all pytorch nn modules
        self.apply(self._init_weights)

    # 权重初始化：用均值0、标准差0.02的正态分布填充权重
    # 目的是给训练一个稳定的起点，避免初始数值太大导致梯度爆炸
    # 训练开始后梯度会自动调整权重，初始化只管第一步不崩
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # c_proj（每个残差分支的最后一层）使用更小的std
            # 原因：残差连接会累积方差
            #   每次残差加法：x = x + c_proj(output)
            #   每个Block有attn和ffn两次残差，N层Block后累积2N次
            #   方差累积：Var(x) = Var(x₀) + 2N × σ²
            #
            # 解决方案：让c_proj初始化更小，抵消层数带来的方差累积
            #   目标：2N × σ² ≈ 常数
            #   推导：σ ∝ 1/sqrt(2N)
            #   结果：std = 0.02 / sqrt(2 * n_layer)
            #
            # 例如12层Block：std = 0.02 / sqrt(24) ≈ 0.004
            # c_proj加入残差流的量只有普通层的1/5，24次累积后方差保持稳定
            if hasattr(module, "RES_CONNECT"):
                # 有个2 因为, 每个Block都会做 2次 残差链接  Attention & Mlp
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, target: Tensor = None):
        B, T = idx.shape
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits: Tensor = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if target is not None:
            # logits: [B, T, vocab_size] → [B*T, vocab_size]，F.cross_entropy要求2维输入
            # target: [B, T] → [B*T]，和logits第一维对齐
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        # hf_keys_set = set(sd_keys_hf)
        # for my in sd_keys:
        #     if my not in hf_keys_set:
        #         print(my)
        # return
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert (
                    sd_hf[k].shape == sd[k].shape
                ), f"mismatched shape: {sd_hf[k].shape} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# ----#

torch.manual_seed(1337)
device = "cpu"
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.mps.is_available():
    torch.mps.manual_seed(1337)
    device = "mps"

num_return_sequences = 5
max_length = 39


# # the hugging face hello world
# from transformers import pipeline, set_seed

# generator = pipeline("text-generation", model="gpt2")
# set_seed(42)
# generator(
#     "Hello, I'm a language model,",
#     max_length=max_length,
#     num_return_sequences=num_return_sequences,
# )


# model = GPT.from_pretrained("gpt2")
model = GPT(GPTConfig())
# model.eval()  # evaluation mode, instead of training mode (performing dropout, BatchNorm  etc in some models)
model.to(device) # model to device will mutate the instance itself
# Triton是torch.compile的依赖，用来在GPU上生成优化的kernel。
# 但是windows 支持很差, 考虑再 Linux 或者 wsl 上运行
# model = torch.compile(model)

#
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5,8)
x = tokens.to(device)


# def create_samples(B: int = 4, T: int = 32):
#     enc = tiktoken.get_encoding("gpt2")
#     with open("input.txt", "r", encoding="utf-8") as f:
#         text = f.read()
#     data = text[:1000]  # first 1,000 characters
#     # print(data[:100])
#     tokens = enc.encode(data)
#     buf = torch.tensor(tokens[: B * T + 1])
#     x = buf[:-1].view(B, T)
#     y = buf[1:].view(B, T)
#     print(x)
#     print(y)
#     return x, y


class DataLoaderLite:
    def __init__(self, B: int, T: int):
        self.B = B
        self.T = T
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        self.token_length = len(self.tokens)
        print(f"loaded {self.token_length} tokens")
        print(f"1 epoch  = {self.tokens // (B *T)} batches")
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T

        if self.current_position + B * T + 1 > self.token_length:
            self.current_position = 0
        return x, y


train_loader = DataLoaderLite(B=16, T=1024)
import time

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 设置float32矩阵乘法的精度模式
# 'high' = 使用TF32格式，牺牲部分尾数精度（23位→10位）换取约8倍乘法速度提升
# 对训练结果影响极小，因为梯度本身就是近似值
# 仅在Nvidia Ampere架构及以上（A100、3090等）有效
# 理论上TF32能提升8倍算力，但GPU做矩阵乘法的瓶颈不只是计算速度，还有把数据从显存搬到计算单元的速度
# 数据底层还是完整的32bit长度, 数据还没搬完计算单元就在等，所以实际只能看到约3倍提升。
torch.set_float32_matmul_precision("high")
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss = model(x, y)
        # import code

        # code.interact(local=locals()) # 类似debugger

    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()  # await gpu
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step={i}, loss={loss.item()}, dt={dt}ms, tok/sec={tokens_per_sec}")


import sys

sys.exit(0)


torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    # no backward annotation
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        # top 50 probability  &  index
        # (5, 50),  (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        ix = torch.multinomial(topk_probs, 1)

        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat((x, xcol), dim=1)


for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens=tokens)
    print(">>>", decoded)
