import torch
import torch.nn.functional as F
import tiktoken
from .train import max_length, num_return_sequences, enc, GPT, detect_device


device = detect_device()
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5,8)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)


model = GPT.from_pretrained("gpt2")
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


for step in range(num_return_sequences):
    tokens = x[step, :max_length].tolist()
    decoded = enc.decode(tokens=tokens)
    print(">>>", decoded)
