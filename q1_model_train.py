import os
import re
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 2  # how many independent sequences will we process in parallel?
block_size = 16  # what is the maximum context length for predictions?
emb_dim = 32    # dimension for vector embedding of each word/token
hidden_size = (512, 1024)  # hidden layer dimensions
max_iters = 300
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
dropout = 0.2
# ------------

# other parameters
max_new_tokens = 50
model_save_path = f"models/next_word_{block_size}_{emb_dim}.pth"
optimizer_save_path = f"models/optimizer_{block_size}_{emb_dim}.pth"

training = False
# ------------

torch.manual_seed(1337)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


def preprocess_text(text):
    """Retain only alphanumeric characters in lowercase"""
    output = ""
    for line in text.split("\n"):
        if line:
            line = re.sub("-", " ", line).lower()
            line = re.sub("[^a-zA-Z0-9. \n]", "", line)
            output += line + " "

    return output


# here are all the unique words that occur in this text
words = ["_", "."]
processed_text = preprocess_text(text).replace(".", "").split()
words.extend(sorted(list(set(processed_text))))
vocab_size = len(words)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(words)}
itos = {i: ch for i, ch in enumerate(words)}


# Tokenizer
def encode(s):
    # encoder: take a string, output a list of integers
    s = preprocess_text(s).split()
    output = []
    for w in s:
        if w.endswith("."):
            word = w[:-1]
            output.append(stoi.get(word, stoi["_"]))
            output.append(stoi["."])
        else:
            output.append(stoi.get(w, stoi["_"]))
    return output


def decode(l):
    # decoder: take a list of integers, output a string
    output = []
    for i in l:
        if itos[i] == ".":
            output[-1] += "."
        else:
            output.append(itos[i])
    return ' '.join(output)


# # Test tokenizer
print("Vocabulary size:", vocab_size)
# t = "To Mr. Sherlock Holmes she is always _the_ adorable woman."
# print(encode(t))
# print(decode(encode(t)))

# # Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split="train"):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class NextWord(nn.Module):

    def __init__(self, vocab_size, block_size=16, emb_dim=32, hidden_size=(512, 1024), act_func='relu'):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        assert isinstance(hidden_size, tuple), "hidden_size must be a tuple"
        assert len(hidden_size) >= 2, "hidden_size must have at least 2 elements"
        assert all(isinstance(x, int) and x >
                   0 for x in hidden_size), "all elements in hidden_size must be positive integers"

        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size[0])
        self.lin2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.lin3 = nn.Linear(hidden_size[1], vocab_size)
        self.act_func = getattr(F, act_func)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx (B,T) and targets (B,) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        tok_emb = tok_emb.view(B, -1)  # Flatten to (B, T*C)

        x = self.act_func(self.lin1(tok_emb))  # (B, h0)
        x = self.act_func(self.lin2(x))  # (B, h1)
        logits = self.lin3(x)  # (B, vocab_size)

        print(logits.size())

        if targets is None:
            loss = None
        else:
            logits = logits.view(B, -1)
            targets = targets.view(B)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Ensure idx only contains the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            # logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = NextWord(vocab_size, block_size, emb_dim, hidden_size)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


if os.path.exists(model_save_path) and os.path.exists(optimizer_save_path):
    model.load_state_dict(torch.load(model_save_path))
    optimizer.load_state_dict(torch.load(optimizer_save_path))
    print("Model and optimizer loaded successfully")

if training:
    model.train()
    elapsed_time = []
    for iter in range(max_iters):
        start_time = time.time()

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')
        # xb is (B,T) and yb is (B,) tensor of integers

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        end_time = time.time()
        elapsed_time.append(end_time - start_time)
    else:
        torch.save(model.state_dict(), model_save_path)
        torch.save(optimizer.state_dict(), optimizer_save_path)

        print(f"Model saved to {model_save_path}")
        print(f"Optimizer saved to {optimizer_save_path}")


# Generate from the model
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens)[0].tolist()))
