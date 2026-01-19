from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch's LayerNorm always has a bias, but GPT-2 uses none. """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # ensure embedding dimension is divisible by number of heads
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)  # linear layer to project input to key, query, value
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)  # linear layer to project output back to embedding dimension
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming convention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))  # lower triangular matrix for causal masking

    def forward(self, x):
        # an efficient implementation of causal multi-head self-attention

        B, T, C = x.size()  # batch size, sequence length, embedding dimension(n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh -> number of heads, hs -> head size, C -> (number of channels) = nh * hs
        # e.g. in GPT-2 (124M) -> nh = 12, hs = 64, C = 768
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)  # split the projections into query, key, value
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs) -> we're making nh(number of heads) into a batch dimension
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs) -> PyTorch treats B and nh as batches, and applies all the subsequent opperations in parallel (in both Batch and number of heads)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # scaled dot-product attention
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # apply autoregressive causal mask to prevent attending to future tokens
        att = F.softmax(att, dim=-1)  # softmax to get attention weights that sum up to 1 (normalized)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side -> performing a concatenation operation
        # output projection
        y = self.c_proj(y) # project back to embedding dimension
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # first linear layer
        self.gelu = nn.GELU(approximate='tanh')  # GELU activation function
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # second linear layer

    def forward(self, x):
        x = self.c_fc(x)  # apply first linear layer
        x = self.gelu(x)  # apply GELU activation
        x = self.c_proj(x)  # apply second linear layer
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # input x goes through layer norm before attention and MLP, following the Pre-LN Transformer architecture -> improves training stability
        # this is different from the original Transformers architecture which uses Post-LN, but is in line with GPT-2 and GPT-3 implementations
        x = x + self.attn(self.ln_1(x))  # input x goes through layer norm, then attention, then residual connection
        x = x + self.mlp(self.ln_2(x))  # input x goes through layer norm, then MLP, then residual connection
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens in the vocabulary: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of transformer layers (blocks)
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension
    dropout: float = 0.0  # dropout rate
    bias: bool = True  # whether to use bias in LayerNorm and Linear layers. True: bias in Linears and Layernorms, like GPT-2. False: no bias, a bit better and faster
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(  # module that allows you to index the submodules by keys, just like dictionary
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embedding weights
            wpe = nn.Embedding(config.block_size, config.n_embd),  # positional encoding weights
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # hidden layers, a list of Block modules
            ln_f = LayerNorm(config.n_embd, bias=config.bias),  # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # language modeling head (classifier)

        # weight tying scheme: the token embedding weights and the lm_head weights are the same
        self.lm_head.weight = self.transformer.wte.weight
        # this improves performance slightly and reduces the number of parameters (by nearly 30% !)

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_embd = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_embd)  # WHY? ("apply dropout to the sum of token and position embeddings")
        for block in self.transformer.h:
            x = block(x)  # pass through each transformer block
        x = self.transformer.ln_f(x)  # final layer norm

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)  # (b, t, vocab_size)
            # compute cross-entropy loss, ignoring padding index -1
            # reshape logits from (b, t, vocab_size) to (b*t, vocab_size), and reshape targets from (b, t) to (b*t)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using x[:, [-1], :] to keep the time dimension
            loss = None
        
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head, n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M parameters
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M parameters
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1.5B parameters
        }[model_type]
        print("forcing vocab_size and block_size to GPT-2 standard values: vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50,257 for GPT-2
        config_args['block_size'] = 1024  # always 1024 for GPT-2
        config_args['bias'] = True  # GPT-2 uses bias in the lm_head
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()  # create the state_dict for our model & Huggingface model
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # ignore these, just a mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']  # weights that need to be transposed due to different conventions
        # the OpenAI checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear. This means that we have to transpose these weights when we import them.
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B  # batch size
        self.T = T  # sequence length

        # at init, load tokens from disk & store them in memory
        with open("data/tiny_shakespeare/input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens from dataset")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches(iterations)")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]  # buffer of (B*T + 1) tokens (+1 for targets)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset to beginning
        if self.current_position + (B + T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


""" Uncomment to run generation example"""
# num_return_sequences = 10
# max_length = 150

# # model = GPT.from_pretrained('gpt2')
# model = GPT(config=GPTConfig())
# print("Model loaded successfully.")

# model.eval()
# model.to("cuda")

# import tiktoken
# enc = tiktoken.get_encoding("gpt2")

# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (num_return_sequences, 8) -> (5, 8)
# x = tokens.to("cuda")

# # generate! right now x is (B, T) where B is 5, T is 8
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     logits = model(x)[0]  # (B, T, vocab_size)
#     # take the logits from the last time step
#     logits = logits[:, -1, :]  # (B, vocab_size)
#     # get the probabilities
#     probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
#     # do top-k sampling of 50 (huggingface piepeline default)
#     # top-k probs here becomes (5, 50), top-k indices becomes (5, 50)
#     topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  # (B, k)
#     # select a token from the top-k probabilities
#     ix = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
#     # gather the corresponding indicies
#     xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
#     # append to the sequence
#     x = torch.cat((x, xcol), dim=1)  # (B, T+1)

# # decode and print all the sequences
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)


"""Uncomment to run a update example"""
# # get a data batch
# import tiktoken
# enc = tiktoken.get_encoding("gpt2")

# with open("data/tiny_shakespeare/input.txt", "r", encoding="utf-8") as f:
#     text = f.read()
# text = text[:1000]
# tokens=  enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])  # (B*T + 1,)
# buf = buf.to("cuda")
# x = buf[:-1].view(B, T)  # (B, T), input
# y = buf[1:].view(B, T)   # (B, T), target

# # get logits
# model = GPT(config=GPTConfig())
# model.to("cuda")

# # optimize
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# for i in range(50):
#     optimizer.zero_grad()  # clear out previous gradients
#     logits, loss = model(x, y)  # forward pass
#     loss.backward()  # backward pass
#     optimizer.step()  # update the parameters
#     print(f"step {i+1}, loss: {loss.item():.5f}")

# import sys; sys.exit()

"""Uncomment to run a data loader example"""
data_loader = DataLoaderLite(B=4, T=32)

model = GPT(config=GPTConfig())
model.to("cuda")

# optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(1000):
    x, y = data_loader.next_batch()
    x, y = x.to("cuda"), y.to("cuda")
    optimizer.zero_grad()  # clear out previous gradients
    logits, loss = model(x, y)  # forward pass
    loss.backward()  # backward pass
    optimizer.step()  # update the parameters
    print(f"step {i+1}, loss: {loss.item():.5f}")

# import sys; sys.exit()


num_return_sequences = 5
max_length = 150

model.eval()
model.to("cuda")

enc = tiktoken.get_encoding("gpt2")

tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (num_return_sequences, 8) -> (5, 8)
x = tokens.to("cuda")

# generate! right now x is (B, T) where B is 5, T is 8
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    logits = model(x)[0]  # (B, T, vocab_size)
    # take the logits from the last time step
    logits = logits[:, -1, :]  # (B, vocab_size)
    # get the probabilities
    probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
    # do top-k sampling of 50 (huggingface piepeline default)
    # top-k probs here becomes (5, 50), top-k indices becomes (5, 50)
    topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  # (B, k)
    # select a token from the top-k probabilities
    ix = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
    # gather the corresponding indicies
    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
    # append to the sequence
    x = torch.cat((x, xcol), dim=1)  # (B, T+1)

# decode and print all the sequences
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)