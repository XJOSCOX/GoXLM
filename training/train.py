from pathlib import Path
from dataclasses import asdict
import hashlib
import json

import torch
import torch.nn.functional as F

from model.transformer import GoXLM, GoXLMConfig


torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load our locally approved experimental data.
data_path = Path("data/approved/science_technology_v01.txt")
raw = data_path.read_bytes()

# Byte-level tokenizer: token IDs from 0 to 255.
tokens = torch.tensor(list(raw), dtype=torch.long)

config = GoXLMConfig(
    vocab_size=256,
    context_length=128,
    n_layers=4,
    n_heads=4,
    n_embd=256,
    dropout=0.1,
)

# A simple split to test the pipeline.
split = int(0.9 * len(tokens))
train_data = tokens[:split]
val_data = tokens[split:]

block_size = 64
batch_size = 16
max_steps = 300

if len(train_data) <= block_size or len(val_data) <= block_size:
    raise ValueError("Dataset is too small for the chosen block size")

model = GoXLM(config).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01,
)


def get_batch(data):
    ix = torch.randint(
        0, len(data) - block_size, (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size] for i in ix
    ]).to(device)

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in ix
    ]).to(device)

    return x, y


@torch.no_grad()
def evaluate():
    model.eval()
    losses = []

    for _ in range(10):
        x, y = get_batch(val_data)
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


print("Starting GoXLM training")
print("Device:", device)
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))
print("Parameters:", sum(p.numel() for p in model.parameters()))

model.train()

for step in range(1, max_steps + 1):
    x, y = get_batch(train_data)

    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step == 1 or step % 50 == 0:
        val_loss = evaluate()
        print(
            f"Step {step:04d} | "
            f"Train loss: {loss.item():.4f} | "
            f"Validation loss: {val_loss:.4f}",
            flush=True
        )

Path("checkpoints").mkdir(exist_ok=True)

checkpoint_path = "checkpoints/goxlm_v01.pt"

torch.save({
    "model": model.state_dict(),
    "config": asdict(config),
    "optimizer": optimizer.state_dict(),
    "step": max_steps,
    "data_sha256": hashlib.sha256(raw).hexdigest(),
}, checkpoint_path)

print("Training completed")
print("Checkpoint:", checkpoint_path)
