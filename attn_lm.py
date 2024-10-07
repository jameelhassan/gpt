import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# hyperparams
bs = 16
learning_rate = 1e-3
eval_iters = 10000
context_size = 8
n_embd = 32

# Load data
txt_pth = './shakespeare_train.txt'
with open(txt_pth, 'r', encoding='utf-8') as f:
    text = f.read()

# Vocabulary of dataset
chars = sorted(list(set(text)))
vocab_size = (len(chars))
print("#### Number of characters and the characters #####")
print(vocab_size, ''.join(chars))

# Mapping functions for characters
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[l] for l in s]
decode = lambda x: ''.join([itos[c] for c in x])

# Data Loading and split to train, val test
data = torch.tensor(encode(text))
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split='train', batch_size=16, context_size=8):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i + context_size] for i in idxs])
    y = torch.stack([data[i + 1:i + 1 + context_size] for i in idxs])
    return x, y

class Attention_Block(nn.Module):
    def __init__(self, n_embd=32):
        super().__init__()
        self.n_embd = n_embd
        self.Wq = nn.Linear(self.n_embd, self.n_embd)
        self.Wk = nn.Linear(self.n_embd, self.n_embd)
        self.Wv = nn.Linear(self.n_embd, self.n_embd)
        self.register_buffer('tril', torch.tril(torch.ones((context_size, context_size))))

    def forward(self, x):
        B, T, C = x.shape
        q = self.Wq(x)  # (B, T, C)
        k = self.Wk(x)  # (B, T, C)
        v = self.Wv(x)  # (B, T, C)
        attn = (q @ k.transpose(1, 2)) * n_embd**(-0.5)    # (B, T, T)
        # Masked attention
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        out = F.softmax(attn, dim=-1) @ v   # (B,T,T) @ (B,T,C) --> (B, T, C)
        return out

# the bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, n_embd=32):
        super().__init__()
        self.n_embd = n_embd
        self.embedding_table = nn.Embedding(vocab_size, self.n_embd)    # (V, C)
        self.positional_embedding = nn.Embedding(context_size, self.n_embd) # (T, C)
        self.attn_block = Attention_Block(n_embd=32)
        self.lm_head = nn.Linear(self.n_embd, vocab_size)   # (C, V)

    def forward(self, idx, target=None):
        # idx, target are both (B,T) tensors of integers
        embeddings = self.embedding_table(idx) # (B,T,C)
        pos_embeddings = self.positional_embedding(torch.arange(idx.shape[1]))  # (T, C)
        x = embeddings + pos_embeddings # broadcast addition (B, T, C)
        x = self.attn_block(x)
        logits = self.lm_head(x)    # (B, T, V)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            loss = F.cross_entropy(logits, target.view(B*T)) # Cross entropy performs softmax on embedding within

        return logits, loss
    
    def generate(self, idx, max_context_length=100):
        # idx - (B, T)
        for i in range(max_context_length):
            logits, loss = self(idx[:, -context_size:]) # (B, T, C)
            # For Bigram model
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, -1) # (B, C)
            # Sampling from the probability distribution to generate next token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# Train a model
lm = BigramLanguageModel(n_embd=n_embd)
optimizer = torch.optim.AdamW(lm.parameters(), lr=learning_rate)

# Training loop
for iter in range(eval_iters):
    # Get batch
    x, y = get_batch('train', batch_size=bs, context_size=context_size)

    # Forward for loss computation
    logits, loss = lm(x, y)

    # Update model
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Print loss logs
    if iter % 500 == 0:
        print(loss.item())

# Thou Generate shakespear from trained model
lm.eval()
inp_token = torch.zeros((1, 1), dtype=torch.long)
print(decode(lm.generate(inp_token, max_context_length=300)[0].tolist()))