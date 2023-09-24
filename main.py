import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

toks = ["hello", "world"]

decode = lambda inp: " ".join([toks[t] for t in inp])

class BigramLM(nn.Module):
    def __init__(self, vocab_len):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_len, vocab_len)
    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits  = self.token_emb_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T) # (B*T := -1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            print(probs, probs.shape)
            # idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            _, idx_next = torch.max(logits, dim=-1, keepdim=True)  # Take the argmax instead of sampling
            print(idx_next, idx_next.shape)
            idx = torch.cat((idx, idx_next), dim=1) # (B, 1+1)
            print(idx, idx.shape)
        return idx

# Initialise Bigram model
model = BigramLM(vocab_len=2)

# Set embedding values
model.token_emb_table.weight = \
    nn.Parameter(torch.tensor([[0., 1.], [0., 0.]], dtype=None))

# Print embedding table
print(model.token_emb_table.weight)

pred = model.generate(torch.tensor([[0]], dtype=torch.long), max_new_tokens=1)
pred_p = pred[0, :]
print("PRED:", pred_p, pred_p.shape)
print(decode(pred_p))