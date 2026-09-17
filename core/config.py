"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from munch import Munch

REQUIRED_SECTIONS = {"model", "training", "loss", "paths", "logging"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration and validate its top-level shape."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping.")

    missing = REQUIRED_SECTIONS.difference(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {', '.join(sorted(missing))}")
    return config


def solver_args(
    config: dict[str, Any],
    *,
    mode: str,
    device: str | None = None,
    input_dir: str | None = None,
    output_dir: str | None = None,
    checkpoint: int | None = None,
    total_iterations: int | None = None,
    batch_size: int | None = None,
) -> Munch:
    """Translate the documented YAML schema into the legacy solver names."""
    model = config["model"]
    training = config["training"]
    loss = config["loss"]
    paths = config["paths"]
    logging = config["logging"]

    args = Munch(
        experiment_name=config.get("experiment_name", "raingan"),
        seed=int(config.get("seed", 777)),
        deterministic=bool(config.get("deterministic", True)),
        device=device or str(config.get("device", "auto")),
        mode=mode,
        img_size=int(model["image_size"]),
        num_domains=int(model["num_domains"]),
        latent_dim=int(model["latent_dim"]),
        hidden_dim=int(model.get("hidden_dim", 512)),
        style_dim=int(model["style_dim"]),
        w_hpf=float(model.get("high_pass_weight", 0)),
        target_domain=int(model.get("target_domain", 1)),
        lambda_reg=float(loss["r1"]),
        gt=float(loss["identity"]),
        lambda_sty=float(loss["style"]),
        lambda_ds=float(loss["diversity"]),
        lambda_npmi=float(loss["npmi"]),
        rain_streak_guidance=bool(loss.get("rain_streak_guidance", True)),
        rain_streak_guidance_weight=float(loss.get("rain_streak_guidance_weight", 0.1)),
        ds_iter=int(loss["diversity_decay_iterations"]),
        randcrop_prob=float(training["random_crop_probability"]),
        total_iters=int(total_iterations or training["total_iterations"]),
        resume_iter=int(checkpoint if checkpoint is not None else training["resume_iteration"]),
        batch_size=int(batch_size or training["batch_size"]),
        val_batch_size=int(training["validation_batch_size"]),
        lr=float(training["learning_rate"]),
        f_lr=float(training["mapping_learning_rate"]),
        beta1=float(training["beta1"]),
        beta2=float(training["beta2"]),
        weight_decay=float(training["weight_decay"]),
        num_workers=int(training["num_workers"]),
        num_outs_per_domain=int(logging["outputs_per_image"]),
        train_img_dir=str(paths["train"]),
        val_img_dir=str(paths["validation"]),
        input_dir=str(input_dir or paths["inference"]),
        checkpoint_dir=str(paths["checkpoints"]),
        sample_dir=str(paths["samples"]),
        out_dir=str(output_dir or paths["outputs"]),
        eval_dir=str(paths["evaluation"]),
        print_every=int(logging["print_every"]),
        sample_every=int(logging["sample_every"]),
        save_every=int(logging["save_every"]),
    )
    if args.img_size < 32 or args.img_size & (args.img_size - 1):
        raise ValueError("model.image_size must be a power of two and at least 32.")
    if not 0 <= args.target_domain < args.num_domains:
        raise ValueError("model.target_domain must be within [0, num_domains).")
    if args.batch_size <= 0 or args.val_batch_size <= 0:
        raise ValueError("Training and validation batch sizes must be positive.")
    if args.total_iters <= 0 or args.resume_iter < 0:
        raise ValueError("Iteration counts must be positive, with resume_iteration >= 0.")
    if args.ds_iter <= 0:
        raise ValueError("loss.diversity_decay_iterations must be positive.")
    if not 0 <= args.randcrop_prob <= 1:
        raise ValueError("training.random_crop_probability must lie in [0, 1].")
    if args.num_workers < 0 or args.num_outs_per_domain <= 0:
        raise ValueError("Worker count cannot be negative and output count must be positive.")
    return args
