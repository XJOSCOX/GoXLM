import torch

from model.transformer import GoXLM, GoXLMConfig

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Only load checkpoints you created or trust.
checkpoint = torch.load(
    "checkpoints/goxlm_v01.pt",
    map_location=device,
    weights_only=True
)

config = GoXLMConfig(**checkpoint["config"])

model = GoXLM(config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

print("=" * 50)
print("GoXLM-0.1 TEXT GENERATION")
print("=" * 50)
print("Model loaded successfully")
print("Training steps:", checkpoint["step"])
print("Device:", device)

prompt = "Science is"

# Our first tokenizer maps UTF-8 bytes to integers.
tokens = list(prompt.encode("utf-8"))
idx = torch.tensor([tokens], dtype=torch.long, device=device)

temperature = 0.8
top_k = 40
max_new_tokens = 200

with torch.inference_mode():
    for _ in range(max_new_tokens):
        context = idx[:, -config.context_length:]

        logits, _ = model(context)

        next_logits = logits[:, -1, :] / temperature

        values, _ = torch.topk(next_logits, k=top_k)
        threshold = values[:, -1].unsqueeze(-1)

        next_logits = next_logits.masked_fill(
            next_logits < threshold,
            float("-inf")
        )

        probabilities = torch.softmax(next_logits, dim=-1)

        next_token = torch.multinomial(
            probabilities, num_samples=1
        )

        idx = torch.cat([idx, next_token], dim=1)

output = bytes(idx[0].tolist()).decode(
    "utf-8", errors="replace"
)

print()
print("Prompt:", prompt)
print()
print("Generated text:")
print(output)
print()
print("=" * 50)
