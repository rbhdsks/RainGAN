#!/usr/bin/env python3
"""Validate RainGAN training pairs or clear inference images and write a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(root: Path, paired: bool) -> dict:
    images = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found below {root}")

    records = []
    invalid = []
    for path in images:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        if paired and width != 2 * height:
            invalid.append(f"{path}: expected width=2*height, found {width}x{height}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "width": width,
                "height": height,
                "sha256": sha256(path),
            }
        )

    if invalid:
        preview = "\n".join(invalid[:10])
        raise ValueError(f"Invalid paired images ({len(invalid)}):\n{preview}")
    return {"root": str(root.resolve()), "paired": paired, "count": len(records), "files": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--paired", action="store_true", help="Require side-by-side clean/rain pairs."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    result = inspect(args.root, args.paired)
    manifest = args.manifest or args.root / "manifest.sha256.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {result['count']} images. Manifest: {manifest}")


if __name__ == "__main__":
    main()
