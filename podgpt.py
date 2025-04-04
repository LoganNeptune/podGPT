import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many indepenent sequences will we process in parellel
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
#-------------

torch.manual_seed(1337)

# wget https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare
with open('pod.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars) }
itos = {i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #encoder take a string, output a list integers
decode = lambda l: ''.join([itos[i] for i in l]) #reverse the map and concatenate

# Train and test splits 
data = torch.tensor(encode(text), dtype=torch.long)
n= int(.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
  # generate a small batch of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
      
class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, 16)
        q = self.query(x) # (B, T, 16)
        #compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, T, 16) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # perform aggregation on the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T)(B,T,C)--->(B,T,C)
        return out

class MultiHeadAttention(nn.Module):
   """multiple heads of self attention in parralel"""

   def __init__(self, num_heads, head_size):
      super().__init__()
      self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
      
      def forward(self, x):
         return torch.cat([h(x) for h in self.heads], dim=-1)

class FeedForward(nn.Module):
   """" a simple linear layer followed by a non-linearity"""

   def __init__(self, n_embd):
      super().__init__[]
      self.net = nn.Sequential(
         nn.Linear(n_embd, n_embd),
         nn.ReLU(),
      )

      def forward(self, x):
         return self.net(x)

class Block(nn.Module):
   """"Transformer block: communication followed by completion"""

   def __init__(self, n_embd, n_head):
      # n_embd: embedding dimension, n_head: the number of heads we'd like
      super().__init__()
      head_size = n_embd // n_head
      self.sa = MultiHeadAttention(n_head, head_size)
      self.ffwd = FeedForward(n_embd)

      def forward(self, x):
         x = self.sa(x)
         x = self.ffws(x)
      return x
      
# super simple bigram model
class BigramLanguageModule(nn.Module):

  def __init__(self):
      super().__init__()
      # each token directly reads off the logits for the next token from a lookup table
      self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
      self.position_embedding_table = nn.Embedding(block_size, n_embd)
      self.blocks = nn.Sequential(
          Block(n_embd, n_head=4),
          Block(n_embd, n_head=4),
          Block(n_embd, n_head=4),
      )

  def forward(self, idx, targets=None):
    B, T = idx.shape

     # idx and targets are both (B, T) tonsor of integers
    tok_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb # (B,T,C)
    x = self.sa_head(x) # Apply one head of self-attention. (B, T, C)
    x = self.ffwd(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T, vocab_size)

    if targets is None:
        loss = None
    else:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        
    return logits,loss

  def generate(self, idx, max_new_tokens):
   # idx is {B, T} array of indices in the current context
      for _ in range(max_new_tokens):
      # crop idx to the last block_size_tokens
        idx_cond = idx[:, -block_size:]
      # get the predictions
        logits, loss = self(idx_cond)
      # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
      return idx
   
model = BigramLanguageModule()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on the train and val sets
    if iter % eval_interval == 0:
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


block_size = 8
train_data[:block_size+1] #+1 packs 8 examples into chunk of 9 char

x = train_data[:block_size]
y = train_data[1:block_size+1] #1 because y are the targets for each position
for t in range(block_size):
  context = x[:t+1]
  target = y[t]
  print(f"when input is {context} the target is: {target}")

torch.manual_seed(1337)
batch_size = 4 #sequences processing in parellel
batch_size = 8 # maximum prediction context length

def get_batch(split):
  #generate a small batch
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
  for t in range(block_size): # time dimension
    context = xb[b, :t+1]
    target = yb[b,t]
    print(f"when input is {context.tolist()} the target: {target}")

print(xb) #input to the transformer

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    # each token reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

  def forward(self, idx, targets=None):

      # idx and targets are both (B,T) tensor of integers
    logits = self.token_embedding_table(idx) # (B,T,C)
    if targets is None:
        loss = None
    else:#Match Cross-Entropy Function
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # get the predictions
        logits, loss = self(idx)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        #apply softmax to get get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

#Randomly trained
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

#Create a Pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(100):

  # sample a batch of data
  xb, yb = get_batch('train')
  
  # evaluate the loss
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(loss.item())

"""##The mathematical trick in self-attention"""

# consider the following toy example:

torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
x.shape

# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)

# version 2
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T,) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)

# version 3
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)

torch.tril(torch.ones(3, 3))

torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a=') #Muliply row...
print(a)
print('--')
print('b=') #By column...
print(b)
print('--')
print('c=') #for .product
print(c)