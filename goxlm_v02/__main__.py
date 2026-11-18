"""Train, inspect, evaluate and generate with GoXLM 0.2."""
import argparse
from pathlib import Path

import torch

from .common import canonical, verify_baseline, write_json
from .data import load_release, read
from .model import generate
from .run import RunConfig, evaluate, load_checkpoint, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    cmd.add_parser("verify-baseline")
    training = cmd.add_parser("train")
    scoring = cmd.add_parser("evaluate")
    generation = cmd.add_parser("generate")
    for command in (training, scoring):
        command.add_argument("catalog", type=Path)
        command.add_argument("release", type=Path)
    for command in (training, scoring, generation):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    training.add_argument("output", type=Path)
    training.add_argument("--steps", type=int, required=True)
    training.add_argument("--config", type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--checkpoint-every", type=int, default=1)
    for command in (scoring, generation):
        command.add_argument("checkpoint", type=Path)
    scoring.add_argument("--split", choices=("validation", "test"), default="validation")
    scoring.add_argument("--output", type=Path)
    generation.add_argument("prompt")
    generation.add_argument("--catalog", type=Path, required=True)
    generation.add_argument("--steps", type=int, default=40)
    generation.add_argument("--top-k", type=int, default=40)
    generation.add_argument("--temperature", type=float, default=0.8)
    generation.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "verify-baseline":
        print(canonical(verify_baseline()).decode(), end="")
    elif args.command == "train":
        config = RunConfig(**read(args.config)) if args.config else RunConfig()
        print(train(args.catalog, args.release, args.output, args.steps, args.device,
                    config, args.resume, args.checkpoint_every))
    else:
        model, checkpoint = load_checkpoint(args.checkpoint, args.device)
        if args.command == "generate":
            release = args.catalog / "releases" / (checkpoint["dataset_version"] + ".json")
            load_release(args.catalog, release)
            print(generate(model, args.prompt, steps=args.steps, top_k=args.top_k,
                           temperature=args.temperature, seed=args.seed))
        else:
            release = load_release(args.catalog, args.release)
            if checkpoint["dataset_version"] != args.release.stem:
                raise ValueError("Checkpoint and evaluation dataset differ")
            rows = [row for row in release["documents"] if row["split"] == args.split]
            result = {"schema": "goxlm-evaluation-0.2.1", "split": args.split,
                      "dataset_version": args.release.stem, "checkpoint_step": checkpoint["step"],
                      "score": evaluate(model, rows, torch.device(args.device))}
            if args.output:
                write_json(args.output, result)
            print(canonical(result).decode(), end="")


if __name__ == "__main__":
    main()
