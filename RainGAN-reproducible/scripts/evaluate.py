#!/usr/bin/env python3
"""Compute FID and paired LPIPS for generated rainy images."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import functional as TF

SUFFIXES = {".jpg", ".jpeg", ".png"}


def image_paths(root: Path, exclude_names: set[str] | None = None) -> list[Path]:
    exclude_names = exclude_names or set()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUFFIXES and path.name not in exclude_names
    )
    if not paths:
        raise ValueError(f"No images found below {root}")
    return paths


def load_batch(paths: list[Path], size: tuple[int, int], device: torch.device) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
            tensors.append(TF.to_tensor(image))
    return torch.stack(tensors).to(device)


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def compute_fid(real: list[Path], generated: list[Path], batch_size: int, device):
    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for batch in chunks(real, batch_size):
        metric.update(load_batch(batch, (299, 299), device), real=True)
    for batch in chunks(generated, batch_size):
        metric.update(load_batch(batch, (299, 299), device), real=False)
    return float(metric.compute().cpu())


def read_pairs(csv_path: Path, real_root: Path, generated_root: Path):
    pairs = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"real", "generated"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain columns: real, generated")
        for row in reader:
            pairs.append((real_root / row["real"], generated_root / row["generated"]))
    return pairs


def matched_pairs(real_root: Path, generated_root: Path):
    real = {path.relative_to(real_root).as_posix(): path for path in image_paths(real_root)}
    generated = {
        path.relative_to(generated_root).as_posix(): path
        for path in image_paths(generated_root, exclude_names={"clear.png"})
    }
    return [(real[name], generated[name]) for name in sorted(real.keys() & generated.keys())]


def compute_lpips(pairs, batch_size: int, device):
    if not pairs:
        return None
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    for batch in chunks(pairs, batch_size):
        real = load_batch([item[0] for item in batch], (256, 256), device)
        generated = load_batch([item[1] for item in batch], (256, 256), device)
        metric.update(real, generated)
    return float(metric.compute().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/metrics.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    real = image_paths(args.real)
    generated = image_paths(args.generated, exclude_names={"clear.png"})
    pairs = (
        read_pairs(args.pairs_csv, args.real, args.generated)
        if args.pairs_csv
        else matched_pairs(args.real, args.generated)
    )
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_directory": str(args.real.resolve()),
        "generated_directory": str(args.generated.resolve()),
        "real_images": len(real),
        "generated_images": len(generated),
        "paired_images": len(pairs),
        "fid": compute_fid(real, generated, args.batch_size, device),
        "lpips": compute_lpips(pairs, args.batch_size, device),
        "device": str(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
