"""Command-line entry point for RainGAN-Kathmandu experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from munch import Munch

from core.config import load_config, solver_args
from core.data_loader import get_test_loader, get_train_loader
from core.reproducibility import (
    environment_snapshot,
    resolve_device,
    seed_everything,
    write_run_metadata,
)
from core.solver import Solver


def _existing_directory(path: str, label: str) -> None:
    candidate = Path(path)
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"{label} directory not found: {candidate}. "
            "See docs/DATASETS.md for the expected layout."
        )


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    mode = {"train": "train", "sample": "sample", "generate": "syn"}[args.command]
    runtime = solver_args(
        config,
        mode=mode,
        device=args.device,
        input_dir=getattr(args, "input_dir", None),
        output_dir=getattr(args, "output_dir", None),
        checkpoint=getattr(args, "checkpoint", None),
        total_iterations=getattr(args, "total_iterations", None),
        batch_size=getattr(args, "batch_size", None),
    )
    runtime.device = str(resolve_device(runtime.device))
    seed_everything(runtime.seed, runtime.deterministic)

    if mode == "train":
        _existing_directory(runtime.train_img_dir, "Training")
        _existing_directory(runtime.val_img_dir, "Validation")
        domains = sorted(
            path.name for path in Path(runtime.train_img_dir).iterdir() if path.is_dir()
        )
        if len(domains) != runtime.num_domains:
            raise ValueError(
                f"Expected {runtime.num_domains} training-domain directories below "
                f"{runtime.train_img_dir}, found {len(domains)}: {domains}"
            )
        metadata_dir = runtime.checkpoint_dir
    else:
        _existing_directory(runtime.input_dir, "Input")
        if runtime.resume_iter <= 0:
            raise ValueError("Generation and sampling require --checkpoint greater than zero.")
        metadata_dir = runtime.out_dir if mode == "syn" else runtime.sample_dir

    resolved = {"source_config": str(Path(args.config).resolve()), "runtime": dict(runtime)}
    metadata_path = write_run_metadata(metadata_dir, resolved)
    print(f"Run metadata: {metadata_path}")
    print(f"Device: {runtime.device}")

    solver = Solver(runtime)
    if mode == "train":
        loaders = Munch(
            src=get_train_loader(
                root=runtime.train_img_dir,
                which="source",
                img_size=runtime.img_size,
                batch_size=runtime.batch_size,
                prob=runtime.randcrop_prob,
                num_workers=runtime.num_workers,
                seed=runtime.seed,
            ),
            ref=get_train_loader(
                root=runtime.train_img_dir,
                which="reference",
                img_size=runtime.img_size,
                batch_size=runtime.batch_size,
                prob=runtime.randcrop_prob,
                num_workers=runtime.num_workers,
                seed=runtime.seed + 1,
            ),
            val=get_test_loader(
                root=runtime.val_img_dir,
                img_size=runtime.img_size,
                batch_size=runtime.val_batch_size,
                shuffle=False,
                num_workers=runtime.num_workers,
                seed=runtime.seed,
            ),
        )
        solver.train(loaders)
        return

    loader = get_test_loader(
        root=runtime.input_dir,
        img_size=runtime.img_size,
        batch_size=runtime.val_batch_size,
        shuffle=False,
        num_workers=runtime.num_workers,
        seed=runtime.seed,
    )
    loaders = Munch(val=loader)
    if mode == "sample":
        solver.sample(loaders)
    else:
        solver.synthesis(loaders)


def doctor(config_path: str) -> None:
    config = load_config(config_path)
    solver_args(config, mode="train")
    snapshot = environment_snapshot()
    snapshot["resolved_auto_device"] = str(resolve_device("auto"))
    snapshot["config_valid"] = True
    print(json.dumps(snapshot, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate RainGAN-Kathmandu reproducibly."
    )
    parser.add_argument(
        "--config",
        default="configs/paper.yaml",
        help="YAML experiment configuration (default: configs/paper.yaml)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Override the configured compute device.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the rain generator.")
    train.add_argument("--checkpoint", type=int, default=None, help="Resume iteration.")
    train.add_argument("--total-iterations", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)

    sample = subparsers.add_parser("sample", help="Create validation samples.")
    sample.add_argument("--checkpoint", type=int, required=True)
    sample.add_argument("--input-dir", default=None)

    generate = subparsers.add_parser("generate", help="Generate rainy images.")
    generate.add_argument("--checkpoint", type=int, required=True)
    generate.add_argument("--input-dir", default=None)
    generate.add_argument("--output-dir", default=None)

    subparsers.add_parser("doctor", help="Validate configuration and report the environment.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "doctor":
        doctor(args.config)
        return
    run(args)


if __name__ == "__main__":
    main()
