## Making a custom GPT model from scratch using PyTorch
## Learned and implemented by following Andrej Karpathy’s “GPT from scratch / nanoGPT” videos
## Core philosophy: tokens in → logits out, minimal but correct Transformer

from transformers import GPT2LMHeadModel, GPT2Tokenizer   ## Used ONLY for loading pretrained GPT-2 weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect
import math


# =============================================================================
# GPT Configuration
# This mirrors GPT-2 style configs exactly, keeping everything explicit
# =============================================================================
@dataclass
class GPTConfig():
    vocab_size : int = 50257        # GPT-2 tokenizer vocab size
    n_embd : int = 768              # embedding dimension (residual stream width)
    block_size : int = 1024         # maximum context length
    n_head : int = 12               # number of attention heads
    n_layers : int = 12             # number of transformer blocks
    dropout : float = 0.0           # GPT-2 uses 0.0 dropout
    bias : bool = True              # GPT-2 uses bias in Linear and LayerNorm


# =============================================================================
# LayerNorm (with optional bias)
# PyTorch does not support bias=False directly, so we implement it ourselves
# =============================================================================
class LayerNorm(nn.Module):
    """LayerNorm with optional bias, matching GPT-2 behavior"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(
            input,
            self.weight.shape,
            self.weight,
            self.bias,
            1e-5
        )


# =============================================================================
# Causal Self-Attention
# This is the "communication" step — tokens talk to each other
# =============================================================================
class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Single Linear layer that produces Q, K, V together (GPT-2 trick)
        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias
        )

        # Output projection back into the residual stream
        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        # Use PyTorch flash attention if available (fast CUDA kernel)
        self.flash = hasattr(
            torch.nn.functional,
            'scaled_dot_product_attention'
        )

        # Manual causal mask fallback (not needed if flash attention exists)
        # self.register_buffer(
        #     'bias',
        #     torch.tril(torch.ones(config.block_size, config.block_size))
        #         .view(1, 1, config.block_size, config.block_size)
        # )

    def forward(self, x):
        B, T, C = x.size()  # batch, time, channels

        # Project once → split into Query, Key, Value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled dot-product attention with causal masking
        # This enforces autoregressive behavior
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            is_causal=True
        )

        # Reassemble heads back into residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + residual dropout
        y = self.residual_dropout(self.c_proj(y))
        return y


# =============================================================================
# MLP Block
# This is the "computation" step — tokens think independently
# =============================================================================
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Expand residual stream by 4x (GPT-2 design)
        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias
        )

        self.gelu = nn.GELU()

        # Project back down to residual size
        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# =============================================================================
# Transformer Block (Pre-Norm)
# ln → attention → residual → ln → mlp → residual
# =============================================================================
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn_c = CasualSelfAttention(config)  # communication
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp_c = MLP(config)                   # computation

    def forward(self, x):
        x = x + self.attn_c(self.ln_1(x))
        x = x + self.mlp_c(self.ln_2(x))
        return x


# =============================================================================
# GPT Model
# =============================================================================
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # All names match GPT-2 exactly to allow weight loading
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # positional embeddings
            dropout = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias)
        ))

        # Language modeling head
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # Weight tying: output projection shares weights with token embedding
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)


    # GPT-2 style initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    # Forward pass: tokens → logits (+ optional loss)
    def forward(self, idx, targets=None):
        B, T = idx.size()
        device = idx.device

        # Enforce context window
        assert T <= self.config.block_size, \
            f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        # Positional indices
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Token + positional embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # During generation, only last token logits are needed
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


    # =============================================================================
    # Load pretrained GPT-2 weights from HuggingFace into this scratch model
    # =============================================================================
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)

        print(f"loading weights from pretrained gpt: {model_type}")

        # Model-specific configs
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        # GPT-2 constants
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True

        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']

        # Initialize scratch model
        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        # Load HF model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Ignore HF attention buffers
        sd_keys_hf = [
            k for k in sd_hf.keys()
            if not k.endswith('.attn.bias') and not k.endswith('.attn.masked_bias')
        ]

        # HF uses Conv1D — need to transpose weights
        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]

        assert len(sd_keys_hf) == len(sd_keys)

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                sd[k].copy_(sd_hf[k].t())
            else:
                sd[k].copy_(sd_hf[k])

        return model


    # =============================================================================
    # Optimizer configuration (AdamW with correct weight decay handling)
    # =============================================================================
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {
            pn: p for pn, p in self.named_parameters() if p.requires_grad
        }

        # Weight decay only on matrix weights (not bias / LayerNorm)
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        non_decay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay_params, 'weight_decay': 0.0}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )

        return optimizer


    # =============================================================================
    # Autoregressive text generation
    # =============================================================================
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.1, top_k=None):
        for _ in range(max_new_tokens):

            # Crop context if it exceeds block size
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )

            logits, _ = self(idx_cond)

            # Focus only on last token
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=-1)

        return idx
