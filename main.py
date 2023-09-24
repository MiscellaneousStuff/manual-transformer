import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

block_size = 3
vocab_len  = 3

toks = ["hello", "world", "geez"]
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

def bigram():
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

class Head(nn.Module):
    def __init__(self, head_size, n_embed):
        super().__init__()
        self.key   = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape

        print("HEAD INP:", x)

        k = self.key(x)   # (B, T, C)
        print("KEY:", k)
        q = self.query(x) # (B, T, C)
        print("QUERY:", q)

        # compute attention scores "affinities"
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        print("Q * K:", wei)

        # perform weighted aggregation of values
        v   = self.value(x)
        print("VAL:", v)
        out = wei @ v

        print("(Q * K) * V:", out)

        return out

class ImprovedLM(nn.Module):

    def __init__(self, vocab_len, block_size, head_size, n_embed, device):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_len, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(head_size=n_embed, n_embed=n_embed) # self-attn head
        self.lm_head = nn.Linear(n_embed, vocab_len, bias=False)
        self.device = device

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape

        token_embed = self.token_emb_table(idx) # (B, T, C)

        print("STR FWD" + "=" * 40)
        print("TOKEN_EMBED:", token_embed)
        pos_emb     = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        print("POS_EMB:", pos_emb)
        x           = token_embed + pos_emb
        print("TOKEN + POS:", x)
        x           = self.sa_head(x) # apply one head of self-attn (B, T, C)
        print("SA HEAD:", x, x.shape)
        # print("sa_head.shape", x.shape)
        logits      = self.lm_head(x) # (B, T, vocab_size)
        print("LM LOGITS:", logits)
        # print("lm_head.shape", x.shape)
        print("END FWD" + "=" * 40)

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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            print("PROBS:", probs)
            #idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            _, idx_next = torch.max(probs, dim=-1, keepdim=True)  # Take the argmax instead of sampling
            idx = torch.cat((idx, idx_next), dim=1) # (B, 1+1)
        return idx
    
def head():
    # Initialse Self-Attention Head Model
    model = ImprovedLM(
        block_size=block_size,
        head_size=3,
        n_embed=3,
        vocab_len=vocab_len,
        device="cpu")
    
    # Set token embedding values
    model.token_emb_table.weight = \
        nn.Parameter(torch.tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]], dtype=None))

    # Set position embedding
    # model.position_embedding_table.weight = \
    #     nn.Parameter(torch.zeros(*model.position_embedding_table.weight.shape, dtype=None))

    # model.position_embedding_table.weight = \
    #     nn.Parameter(torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
    model.position_embedding_table.weight = \
        nn.Parameter(torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]))

    # Set head k, q, v weight values
    with torch.no_grad():
        # model.sa_head.query.weight.copy_(torch.ones_like(model.sa_head.query.weight))
        # model.sa_head.key.weight.copy_(torch.ones_like(model.sa_head.key.weight))
        # model.sa_head.value.weight.copy_(torch.ones_like(model.sa_head.value.weight))
        model.sa_head.query.weight.copy_(torch.eye(model.sa_head.query.weight.size(0)))
        model.sa_head.key.weight.copy_(torch.eye(model.sa_head.key.weight.size(0)))
        model.sa_head.value.weight.copy_(torch.eye(model.sa_head.value.weight.size(0)))

    # Set MLP weight values
    with torch.no_grad():
        # model.lm_head.weight.copy_(torch.ones_like(model.lm_head.weight))
        model.lm_head.weight.copy_(torch.eye(model.sa_head.value.weight.size(0)))

    print("model.token_emb_table.weight:", model.token_emb_table.weight, model.token_emb_table.weight.shape)
    print("model.position_embedding_table.weight:", model.position_embedding_table.weight, model.position_embedding_table.weight.shape)
    print("model.sa_head.query.weight:", model.sa_head.query.weight, model.sa_head.query.weight.shape)
    print("model.sa_head.key.weight:", model.sa_head.key.weight, model.sa_head.key.weight.shape)
    print("model.sa_head.value.weight:", model.sa_head.value.weight, model.sa_head.value.weight.shape)
    print("model.lm_head.weight:", model.lm_head.weight, model.lm_head.weight.shape)
    
    # Pred
    pred = model.generate(torch.tensor([[0]], dtype=torch.long), max_new_tokens=2)
    pred_p = pred[0, :]
    print("PRED:", pred_p, pred_p.shape)
    print(decode(pred_p))

if __name__ == "__main__":
    # bigram()
    head()