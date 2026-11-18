"""Explicit entry point for the historical GoXLM-0.1 experiment."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

from goxlm_v02.common import ROOT, git_provenance, publish, source_fingerprint


DATA_PATH = ROOT / "data/approved/science_technology_v01.txt"
MAX_STEPS = 300


def train(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Checkpoint path already exists: {output}")

    import torch

    from model.transformer import GoXLM, GoXLMConfig

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = DATA_PATH.read_bytes()
    tokens = torch.tensor(list(raw), dtype=torch.long)
    config = GoXLMConfig(
        vocab_size=256,
        context_length=128,
        n_layers=4,
        n_heads=4,
        n_embd=256,
        dropout=0.1,
    )
    split = int(0.9 * len(tokens))
    train_data = tokens[:split]
    val_data = tokens[split:]
    block_size = 64
    batch_size = 16
    if len(train_data) <= block_size or len(val_data) <= block_size:
        raise ValueError("Dataset is too small for the chosen block size")

    provenance = {"git": git_provenance(),
                  "code_revision": source_fingerprint(ROOT, [
                      ROOT / "model/transformer.py",
                      ROOT / "goxlm_v02/common.py",
                      Path(__file__).resolve(),
                  ])}
    model = GoXLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    def get_batch(data):
        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
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
    for step in range(1, MAX_STEPS + 1):
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
                flush=True,
            )

    checkpoint = {
        "model": model.state_dict(),
        "config": asdict(config),
        "optimizer": optimizer.state_dict(),
        "step": MAX_STEPS,
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        **provenance,
    }
    publish(output, lambda stream: torch.save(checkpoint, stream))
    print("Training completed")
    print("Checkpoint:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="write to a new checkpoint file")
    args = parser.parse_args()
    try:
        train(args.output)
    except FileExistsError as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
