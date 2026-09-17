# Reproduction protocol

## What a complete run must retain

For every training or generation run, archive these together:

1. Git commit hash.
2. `configs/paper.yaml` and generated `run_metadata.json`.
3. Training-data `manifest.sha256.json`.
4. All four checkpoint files for a training iteration: networks, EMA networks, optimizer state, and RNG/training state.
5. Generated `manifest.csv`.
6. Metric JSON plus the real/generated pair manifest used by LPIPS.

## Recommended sequence

```bash
python main.py --config configs/paper.yaml doctor
python scripts/download_kathmandu.py --dataset primary
python scripts/validate_data.py data/rains/train --paired
python scripts/validate_data.py data/rains/val
python main.py --config configs/paper.yaml --device cuda train
python main.py --config configs/paper.yaml --device cuda generate --checkpoint 100000
python scripts/evaluate.py --real <real-rain-root> --generated artifacts/generated/raingan_kathmandu
```

## Expected paper configuration

- Image size: 256
- Latent dimension: 16
- Style dimension: 64
- Batch size: 8
- Adam learning rate: 1e-4
- Mapping-network learning rate: 1e-6
- Adam betas: 0.01 and 0.99
- Weight decay: 1e-4
- Training iterations: 100,000
- Seed: 777
- Generated variants per clear image: 5

All values are versioned in `configs/paper.yaml`. Command-line overrides are written to `run_metadata.json`.

## Known boundary

Exact result reproduction is currently blocked by artifacts absent from the original public repository: the paper checkpoint, exact training file inventory/split, real-rain evaluation pairing, and RF-DETR experiment configuration. The refactored code makes future runs traceable, but it cannot reconstruct unrecorded historical randomness or data splits.
