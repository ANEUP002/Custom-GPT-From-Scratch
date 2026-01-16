##Making the custom GPT model from scratch using transformers and pytorch
from transformers import GPT2LMHeadModel, GPT2Tokenizer   ##Importing the weights of the model from the HUB.
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect
import math
#________Defining the configurations for the model__________##
@dataclass
class GPTConfig():
    vocab_size : int = 50257
    n_embd : int = 768
    block_size :int = 1024
    n_head : int = 12
    n_layers: int = 12
    dropout : float = 0.0
    bias : bool = True

##___________Defining our model from here_____________________##
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)



class CasualSelfAttention(nn.Module):
    def __init__(self, config):  
        super().__init__()
        self.config = config
        ##This is the inner projections that we have
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias = config.bias)
        ##This is for the outer projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attn_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional,'scaled_dot_product_attention') ##Use flassh attention which is optimized in CUDA kernels for faster scaled dot product attention and matrix multiplication.
        #if not self.flash:   ##fallback if the flash attention is not available
            #self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size).view(1,1,config.block_size, config.block_size)))
    def forward(self, x):
        B,T,C = x.size()
        q,k,v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        k = k.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        #if self.flash:
        y = torch.nn.functional.scaled_dot_product_attention(q,k,v, is_causal = True)
        #else:
            #attn = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
            #attn = attn.masked_fill(self.bias[:,:,:T,:T] ==0, float('-inf'))
            #attn = F.softmax(attn, dim=-1) ##Always on the last dimension
            #attn = self.attn_dropout(attn)
            #y = attn @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.residual_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):  #MLP for the compute 
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias = config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias = config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias = config.bias)
        self.attn_c = CasualSelfAttention(config) ##Attention for communication
        self.mlp_c = MLP(config) ##MLP for calculatiom
        self.ln_2 = LayerNorm(config.n_embd, bias= config.bias)
    def forward(self, x):
        x = x + self.attn_c(self.ln_1(x))
        x = x + self.mlp_c(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            dropout = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = LayerNorm(config.n_embd, bias= config.bias)

        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.lm_head.weight = self.transformer.wte.weight 
        self.apply(self._init_weights)
    
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self,idx, targets = None):
        B,T = idx.size()
        device = idx.device 
        assert T <= self.config.block_size,f"Cannot forward sequence of length of {T}, block size is of {self.config.block_size} "
        pos = torch.arange(0, T, dtype = torch.long, device = device) 
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.dropout(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index = -1)
        else:
            logits = self.lm_head(x[:,[-1],:])
            loss = None
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict= {pn:p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >=2]
        non_decay_params = [p for n, p in param_dict.items() if p.dim() <2]

        optim_groups = [
            {'params':decay_params, 'weight_decay': weight_decay},
            {'params':non_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        print(f'The number of decayed parameters are {num_decay_params}')
        num_nodecay_params = sum(p.numel() for p in non_decay_params)
        print(f'The number of non decayed paramleters are {num_nodecay_params}')
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        print(f'Using fused Adam{fused_available}')
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr= learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused = use_fused)
        return optimizer


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 0.1, top_k = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:] ##Here we are cropping the sequence length if it gets longer than the block_size
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]/temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx
    








