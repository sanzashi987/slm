from dataclasses import dataclass
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch import Tensor


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # 50000 BPE + 256 bytes token + 1 eof token
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class SelfAttention(nn.Module):
    def __int__(self, config: GPTConfig):
        super.__init__()
        assert config.n_embd & config.n_head == 0
        # q,k,v linear matrix all in one
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        # q,k,v will be a four-dimensional tensor, so the low triangle matrix
        # also needs to be reshaped.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: Tensor):
        B, T, C = x.shape

        qkv = self.c_attn(x)  # (B,T, 3*n_head)
        qkv_spilt: tuple[Tensor, Tensor, Tensor] = torch.split(qkv, dim=2)
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
        att = att.masked_fill(self.trill[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, n_head, T, head_size)
        y = y.transpose(1, 2)  # (B, T, n_head , head_size)
        y = y.contiguous().view(B, T, C)  # (B, T, n_embd)   n_embd = n_head * head_size
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __int__(self, config: GPTConfig):
        super.__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")  # difference to  Relu
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Layer(nn.Module):
    def __init__(self, config: GPTConfig):
        super.__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, input):
        x = x + self.attn(self.ln_1(input))
        x = x + self.mlp(self.ln_2(input))
        return x


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
