#!/usr/bin/env python3
"""Download and verify the public Kathmandu road datasets from Harvard Dataverse."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

DATAVERSE = "https://dataverse.harvard.edu"
DATASETS = {
    "primary": {
        "doi": "doi:10.7910/DVN/YJRUP4",
        "slug": "ranipokhari_narayanhiti",
        "title": "Dataset Kathmandu Road Ranipokhari Narayanhiti",
    },
    "extension": {
        "doi": "doi:10.7910/DVN/38E8AG",
        "slug": "ratnapark_tripureshwor",
        "title": "Kathmandu Road Dataset for Domain Agnostic Tasks",
    },
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "RainGAN-reproducibility/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_metadata(doi: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"persistentId": doi})
    payload = _get_json(f"{DATAVERSE}/api/datasets/:persistentId/?{query}")
    if payload.get("status") != "OK":
        raise RuntimeError(f"Dataverse returned a non-OK response for {doi}")
    return payload["data"]


def safe_relative_path(directory: str | None, label: str) -> Path:
    """Build a local relative path and reject traversal components."""
    parts = []
    if directory:
        parts.extend(PurePosixPath(directory).parts)
    parts.extend(PurePosixPath(label).parts)
    if not parts or any(part in {"", ".", "..", "/"} for part in parts):
        raise ValueError(f"Unsafe Dataverse path: {directory!r}/{label!r}")
    return Path(*parts)


def md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(file_info: dict[str, Any], destination: Path) -> None:
    data_file = file_info["dataFile"]
    expected = data_file.get("checksum", {}).get("value", "").lower()
    if destination.is_file() and expected and md5(destination) == expected:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        f"{DATAVERSE}/api/access/datafile/{data_file['id']}",
        headers={"User-Agent": "RainGAN-reproducibility/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)

    actual = md5(temporary)
    if expected and actual != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {destination.name}: expected {expected}, received {actual}"
        )
    temporary.replace(destination)


def selected_datasets(selection: str) -> list[dict[str, str]]:
    if selection == "all":
        return list(DATASETS.values())
    return [DATASETS[selection]]


def download_dataset(spec: dict[str, str], output: Path, images: Path, list_only: bool) -> None:
    metadata = fetch_metadata(spec["doi"])
    version = metadata["latestVersion"]
    files = version.get("files", [])
    image_files = [f for f in files if Path(f["label"]).suffix.lower() in IMAGE_SUFFIXES]
    print(
        f"{spec['title']}: version {version['versionNumber']}.{version['versionMinorNumber']}, "
        f"{len(files)} files ({len(image_files)} images)"
    )
    if list_only:
        return

    raw_root = output / spec["slug"]
    manifest_files = []
    for index, file_info in enumerate(files, start=1):
        relative = safe_relative_path(file_info.get("directoryLabel"), file_info["label"])
        destination = raw_root / relative
        print(f"[{index:03d}/{len(files):03d}] {relative}")
        download_file(file_info, destination)

        data_file = file_info["dataFile"]
        manifest_files.append(
            {
                "id": data_file["id"],
                "path": relative.as_posix(),
                "size": data_file.get("filesize"),
                "checksum": data_file.get("checksum"),
            }
        )
        if destination.suffix.lower() in IMAGE_SUFFIXES:
            inference_path = images / spec["slug"] / destination.name
            inference_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, inference_path)

    manifest = {
        "doi": spec["doi"],
        "persistent_url": metadata.get("persistentUrl"),
        "title": version.get("datasetTitle"),
        "version": f"{version['versionNumber']}.{version['versionMinorNumber']}",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": manifest_files,
    }
    manifest_path = raw_root / "dataverse_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["primary", "extension", "all"],
        default="primary",
        help="Dataset release to download (default: primary).",
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/kathmandu"))
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/kathmandu/clear"),
        help="Flattened, model-ready clear-image destination.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Fetch metadata and show counts without downloading files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        for spec in selected_datasets(args.dataset):
            download_dataset(spec, args.output, args.images, args.list_only)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
