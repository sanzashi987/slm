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

    def forward(self, idx: Tensor):
        B, T = idx.shape
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

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

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.mps.is_available():
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
model.eval()  # evaluation mode, instead of training mode (performing dropout, BatchNorm  etc in some models)
model.to(device)

#
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5,8)
x = tokens.to(device)


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
