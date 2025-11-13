import torch
import torch.nn.functional as F
import tiktoken
from model import GPT, GPTConfig

# --------------------------
# Setup
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load model
config = GPTConfig(vocab_size=50304)
model = GPT(config).to(device)

# Load weights (ignore compile wrapper prefixes like _orig_mod)
state_dict = torch.load("model.pt", map_location=device)
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
else:
    model.load_state_dict(state_dict, strict=False)

model.eval()
print("✅ Model loaded successfully!")

# --------------------------
# Generation Function
# --------------------------
def generate(
    model,
    start_text="ROMEO:",
    max_new_tokens=800,
    temperature=0.9,
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.15,
):
    model.eval()
    x = torch.tensor([enc.encode(start_text)], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = x[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            for token_id in set(x[0].tolist()):
                logits[0, token_id] /= repetition_penalty

            probs = F.softmax(logits, dim=-1)

            # Apply nucleus (top-p) filtering
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            sorted_probs[mask] = 0
            total_prob = sorted_probs.sum()
            if total_prob.item() == 0 or torch.isnan(total_prob):
                # fallback: if numerical instability, use unfiltered probs
                next_idx = torch.multinomial(probs[0], 1)
            else:
                sorted_probs /= total_prob
                next_idx = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]

            x = torch.cat((x, next_idx.view(1, 1)), dim=1)

    return enc.decode(x[0].tolist())

# --------------------------
# Run Generation
# --------------------------
prompt = input("\n🎭 Enter your Shakespeare-style prompt:\n> ")
output = generate(
    model,
    start_text=prompt,
    max_new_tokens=800,
    temperature=0.85,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.2,
)

print("\n--- 📝 Generated Text ---\n")
print(output)
print("\n--------------------------\n")
