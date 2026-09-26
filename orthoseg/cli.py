"""Command-line interface for source-domain training and target evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .engine import evaluate_experiment, load_config, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orthoseg",
        description="Paper-guided OrthoSeg reference implementation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train one source-domain model")
    train.add_argument("--config", type=Path, required=True, help="JSON experiment configuration")

    evaluate = commands.add_parser("evaluate", help="evaluate without target-domain fine-tuning")
    evaluate.add_argument("--config", type=Path, required=True, help="JSON experiment configuration")
    evaluate.add_argument("--checkpoint", type=Path, required=True, help="model checkpoint")
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "train":
        checkpoint = train_experiment(config)
        print(f"Best validation checkpoint: {checkpoint}")
    else:
        report = evaluate_experiment(
            config, checkpoint_path=args.checkpoint, split=args.split
        )
        print(f"Evaluation report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
