import torch

from model.transformer import GoXLM, GoXLMConfig

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

config = GoXLMConfig()
model = GoXLM(config).to(device)

parameters = sum(p.numel() for p in model.parameters())

print("=" * 45)
print("GoXLM-0.1 NEURAL NETWORK TEST")
print("=" * 45)
print("Device:", device)
print("Parameters:", f"{parameters:,}")
print("Vocabulary:", config.vocab_size)
print("Context length:", config.context_length)

# Synthetic token IDs, NOT external training data.
inputs = torch.randint(
    0, config.vocab_size, (4, 64), device=device
)

targets = torch.randint(
    0, config.vocab_size, (4, 64), device=device
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)

model.train()
logits, loss = model(inputs, targets)

print("Output shape:", tuple(logits.shape))
print("Initial loss:", round(loss.item(), 4))

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Backward pass: SUCCESS")
print("Optimizer update: SUCCESS")

if device == "cuda":
    print(
        "Peak GPU memory:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "GB"
    )

print("=" * 45)
print("GoXLM FIRST NEURAL NETWORK TEST PASSED")
