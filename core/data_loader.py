"""Deterministic datasets and data loaders for RainGAN-Kathmandu."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from munch import Munch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder

from core.reproducibility import seed_worker

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(root: str | Path) -> list[Path]:
    """Return all supported images below ``root`` in a stable order."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root_path}")
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


class ReferenceDataset(data.Dataset):
    """Return two independently ordered paired-image samples from one domain."""

    def __init__(self, root: str | Path, transform=None, seed: int = 777):
        self.samples, self.targets = self._make_dataset(Path(root), seed)
        self.transform = transform

    @staticmethod
    def _make_dataset(root: Path, seed: int):
        rng = random.Random(seed)
        first: list[Path] = []
        second: list[Path] = []
        labels: list[int] = []
        domains = sorted(path for path in root.iterdir() if path.is_dir())
        for index, domain in enumerate(domains):
            domain_images = list_images(domain)
            shuffled = domain_images.copy()
            rng.shuffle(shuffled)
            first.extend(domain_images)
            second.extend(shuffled)
            labels.extend([index] * len(domain_images))
        return list(zip(first, second, strict=False)), labels

    def __getitem__(self, index: int):
        first_path, second_path = self.samples[index]
        first = Image.open(first_path).convert("RGB")
        second = Image.open(second_path).convert("RGB")
        if self.transform is not None:
            first = self.transform(first)
            second = self.transform(second)
        return first, second, self.targets[index]

    def __len__(self) -> int:
        return len(self.targets)


class InferenceDataset(data.Dataset):
    """Load clear images recursively while preserving their relative paths."""

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.samples = list_images(self.root)
        if not self.samples:
            raise ValueError(f"No JPG/JPEG/PNG images found below {self.root}")
        self.transform = transform

    def __getitem__(self, index: int):
        path = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        relative_path = path.relative_to(self.root).as_posix()
        return image, 0, relative_path

    def __len__(self) -> int:
        return len(self.samples)


class DeterministicWeightedSampler(data.Sampler[int]):
    """Draw one weighted sample at a time so sampler RNG can resume exactly."""

    def __init__(self, weights, num_samples: int, generator: torch.Generator):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.generator = generator

    def __iter__(self):
        for _ in range(self.num_samples):
            yield int(
                torch.multinomial(
                    self.weights, 1, replacement=True, generator=self.generator
                ).item()
            )

    def __len__(self):
        return self.num_samples


def _balanced_sampler(labels: list[int], generator: torch.Generator):
    class_counts = np.bincount(labels)
    if not len(class_counts) or np.any(class_counts == 0):
        raise ValueError("Each training domain must contain at least one image.")
    weights = (1.0 / class_counts)[labels]
    return DeterministicWeightedSampler(weights, len(weights), generator)


def _validate_paired_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width != 2 * height:
        raise ValueError(
            "Training pair must contain equal square halves (width=2*height); "
            f"received {width}x{height}."
        )
    return image


def _training_transform(image_size: int, random_crop_probability: float):
    resize = transforms.Resize((image_size, image_size * 2), antialias=True)
    crop = transforms.RandomResizedCrop(
        (image_size, image_size * 2),
        scale=(0.8, 1.0),
        ratio=(1.8, 2.2),
        antialias=True,
    )
    return transforms.Compose(
        [
            transforms.Lambda(_validate_paired_image),
            transforms.Lambda(
                lambda image: crop(image)
                if random.random() < random_crop_probability
                else resize(image)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def get_train_loader(
    root,
    which="source",
    img_size=256,
    batch_size=8,
    prob=0.5,
    num_workers=4,
    seed=777,
):
    print(f"Preparing deterministic {which} training loader from {root}")
    transform = _training_transform(img_size, prob)
    sampler_generator = torch.Generator().manual_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed + 10_000)

    if which == "source":
        dataset = ImageFolder(root, transform)
    elif which == "reference":
        dataset = ReferenceDataset(root, transform, seed=seed)
    else:
        raise ValueError(f"Unknown training loader type: {which}")
    if not dataset:
        raise ValueError(f"No training images found below {root}")

    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=_balanced_sampler(dataset.targets, sampler_generator),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        persistent_workers=num_workers > 0,
    )


def get_test_loader(
    root,
    img_size=256,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    seed=777,
):
    print(f"Preparing deterministic inference loader from {root}")
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    dataset = InferenceDataset(root, transform=transform)
    generator = torch.Generator().manual_seed(seed)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


class InputFetcher:
    """Cycle through loaders and move tensor fields to the selected device."""

    def __init__(self, loader, loader_ref=None, latent_dim=16, mode="", device="cpu"):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.mode = mode
        self.iterator = iter(loader)
        self.reference_iterator = iter(loader_ref) if loader_ref is not None else None

    @staticmethod
    def _next(iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def _fetch_inputs(self):
        batch, self.iterator = self._next(self.iterator, self.loader)
        if len(batch) == 3:
            images, labels, paths = batch
        else:
            images, labels = batch
            paths = None
        return images, labels, paths

    def _fetch_references(self):
        batch, self.reference_iterator = self._next(self.reference_iterator, self.loader_ref)
        return batch

    def __next__(self):
        images, labels, paths = self._fetch_inputs()
        if self.mode == "train":
            references, references_2, reference_labels = self._fetch_references()
            inputs = Munch(
                x_src=images,
                y_src=labels,
                y_ref=reference_labels,
                x_ref=references,
                x_ref2=references_2,
                z_trg=torch.randn(images.size(0), self.latent_dim),
                z_trg2=torch.randn(images.size(0), self.latent_dim),
            )
        elif self.mode == "val":
            inputs = Munch(x_src=images, y_src=labels, paths=list(paths or []))
        else:
            raise ValueError(f"Unknown fetch mode: {self.mode}")

        return Munch(
            {
                key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
        )
