import tiktoken
from model import GPT, GPTConfig
import torch
import time
import math

# ------------------------------
# Data Loader
# ------------------------------
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('shakespeare.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens):,} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]

        # wrap around if we hit the end
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
            buf = self.tokens[: B*T + 1]

        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B*T
        return x, y


# ------------------------------
# Setup
# ------------------------------



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B = 8
T = 1024
grad_accum = total_batch_size // (B*T)


train_loader = DataLoaderLite(B=B, T=T)

# typo fixed: "precison" → "precision"
torch.set_float32_matmul_precision('high')

# You must INSTANTIATE GPTConfig()
model = GPT(GPTConfig(vocab_size=50304)).to(device)
model = torch.compile(model, backend="aot_eager", mode="reduce-overhead")

max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 200

def get_lr(it): ##Scheduler that can be imported from the pytorch but since it is justr 5 lines of code I am defining it manually 
    #Linear Warmup
    if it < warmup_steps:
        return max_lr * (it +1) / warmup_steps
    if it>max_steps:
        return min_lr

    decay_ratio = (it-warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff *(max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device = device)
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas = (0.9,0.95), eps = 1e-8)

# ------------------------------
# Training Loop with Timing
# ------------------------------
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad() 
    loss_accum = 0.0
    for micro_steps in range(grad_accum):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000  # milliseconds
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum) / (t1 - t0)
    print(f"step {step:03d} | loss: {loss_accum.item():.4f}|lr:{lr:.4f}|norm:{norm:.4f} | dt:{dt:.2f} ms | {tokens_per_sec:.2f} tok/s")


    # Save model after training
torch.save(model.state_dict(), "model.pt")
print("✅ Model saved to model.pt")

 