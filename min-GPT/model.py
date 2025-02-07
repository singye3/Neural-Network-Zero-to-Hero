#!/usr/bin/env python
"""
A self-contained autoregressive language model definition with a training loop and tiktoken-based tokenization.
Reads text from an input file ("input.txt").
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
import wandb  


# ----------------------------
# Simple configuration class
# ----------------------------
class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def merge_from_dict(self, d):
        self.__dict__.update(d)
    def __getattr__(self, key):
        # Return None for attributes that haven't been set.
        return None

# ----------------------------
# Causal Self-Attention module
# ----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Ensure embedding size is divisible by the number of heads
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Linear projection for query, key, value (combined)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # Register a lower-triangular mask to ensure causality
        self.register_buffer(
            "bias", torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # Batch, Time, Channels
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Compute scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

# ----------------------------
# Transformer Block
# ----------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict({
            'c_fc': nn.Linear(config.n_embd, 4 * config.n_embd),
            'act': nn.GELU(),  # Using PyTorch's built-in GELU
            'c_proj': nn.Linear(4 * config.n_embd, config.n_embd),
            'dropout': nn.Dropout(config.resid_pdrop)
        })

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        mlp_out = self.mlp['c_fc'](self.ln_2(x))
        mlp_out = self.mlp['act'](mlp_out)
        mlp_out = self.mlp['c_proj'](mlp_out)
        mlp_out = self.mlp['dropout'](mlp_out)
        x = x + mlp_out
        return x

# ----------------------------
# GPT Language Model
# ----------------------------
class GPT(nn.Module):
    @staticmethod
    def get_default_config():
        C = Config(
            model_type='gpt',
            n_layer=None,
            n_head=None,
            n_embd=None,
            vocab_size=None,
            block_size=None,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be set in config"
        assert config.block_size is not None, "block_size must be set in config"
        self.block_size = config.block_size

        # Either specify model_type (which auto-fills n_layer, n_head, n_embd) or provide them directly.
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given, "Either model_type or (n_layer, n_head, n_embd) must be provided."
        if type_given:
            model_configs = {
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }
            config.merge_from_dict(model_configs[config.model_type])

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.embd_pdrop),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        assert len(decay & no_decay) == 0, "Some parameters are in both decay and no_decay sets!"
        assert len(set(param_dict.keys()) - (decay | no_decay)) == 0, "Some parameters were not assigned to any set!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
             "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Sequence length {t} exceeds block size {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer['wte'](idx)
        pos_emb = self.transformer['wpe'](pos)
        x = self.transformer['drop'](tok_emb + pos_emb)
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ----------------------------
# Text Dataset
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx: idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1: idx+1+self.block_size], dtype=torch.long)
        return x, y

# ----------------------------
# Training Loop
# ----------------------------
def train(model, optimizer, dataloader, device, epochs=1, print_every=100):
    model.train()
    step = 0
    for epoch in range(epochs):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

            if step % print_every == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
            step += 1



# ----------------------------
# Main: Prepare data, build model, train, and report metrics to wandb
# ----------------------------
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read text from a file
file_path = "text.txt"  # Update this path as needed
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Input file not found: {file_path}")
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Use tiktoken's GPT-2 encoding for tokenization.
encoding = tiktoken.get_encoding("gpt2")
data = encoding.encode(text)

# Create dataset and dataloader
block_size = 64  # maximum context length
dataset = TextDataset(data, block_size)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Configure the GPT model (using model_type to auto-fill n_layer, n_head, n_embd)
config = Config(
    model_type='gpt-nano',
    vocab_size=encoding.n_vocab,
    block_size=block_size,
    embd_pdrop=0.1,
    resid_pdrop=0.1,
    attn_pdrop=0.1,
)
model = GPT(config)
model.to(device)

# Training hyperparameters configuration
train_config = Config(
    weight_decay=0.1,
    learning_rate=3e-4,
    betas=(0.9, 0.95)
)
optimizer = model.configure_optimizers(train_config)

# Initialize wandb and set up experiment configuration.
wandb.init(project="mini-gpt-language-model", config={
    "model_type": config.model_type,
    "vocab_size": config.vocab_size,
    "block_size": config.block_size,
    "batch_size": 16,
    "learning_rate": train_config.learning_rate,
    "epochs": 20,
    "embd_pdrop": config.embd_pdrop,
    "resid_pdrop": config.resid_pdrop,
    "attn_pdrop": config.attn_pdrop,
})
wandb.watch(model, log="all")  # Optionally log gradients and parameter histograms

# Update the training loop to log metrics to wandb
def train(model, optimizer, dataloader, device, epochs=1, print_every=100):
    model.train()
    step = 0
    for epoch in range(epochs):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

            if step % print_every == 0:
                loss_val = loss.item()
                print(f"Epoch {epoch} | Step {step} | Loss: {loss_val:.4f}")
                wandb.log({"epoch": epoch, "step": step, "loss": loss_val})
            step += 1

# Train the model
print("Starting training...")
train(model, optimizer, dataloader, device, epochs=5, print_every=10)

# Generate text from the trained model
model.eval()
context_text = "And God"
context = torch.tensor([encoding.encode(context_text)], dtype=torch.long).to(device)
generated = model.generate(context, max_new_tokens=50, temperature=1.0, do_sample=True, top_k=10)
output_text = encoding.decode(generated[0].tolist())
print("\nGenerated text:")
print(output_text)

# Log the generated text as a final artifact.
wandb.log({"generated_text": output_text})
