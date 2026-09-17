# Dataset protocol

## 1. Kathmandu clear-road data

The generation stage uses public clear-weather urban road images from Harvard Dataverse.

| Alias | Dataset | Persistent ID | Current API inventory* | Role |
|---|---|---|---:|---|
| `primary` | Dataset Kathmandu Road Ranipokhari Narayanhiti | [doi:10.7910/DVN/YJRUP4](https://doi.org/10.7910/DVN/YJRUP4) | V1.1; 288 files; 143 images | Paper companion / default inference set |
| `extension` | Kathmandu Road Dataset for Domain Agnostic Tasks | [doi:10.7910/DVN/38E8AG](https://doi.org/10.7910/DVN/38E8AG) | V2.1; 314 files; 156 images | Optional second-route extension |

\*Inventory checked through the Dataverse API on 17 September 2026. The primary landing-page description says 146 images while the current API exposes 143 image files. The downloader records the live version and per-file checksums so this discrepancy remains visible rather than being silently ignored.

Run:

```bash
python scripts/download_kathmandu.py --dataset primary
python scripts/download_kathmandu.py --dataset extension
```

Files are preserved under `data/raw/kathmandu/<route>/`. Model-ready copies are placed below `data/kathmandu/clear/<route>/`. Object-detection labels are retained in the raw archive but are not inputs to the GAN.

## 2. GAN training pairs

The Harvard releases provide the clear Kathmandu targets and detection annotations; they do not by themselves reproduce the GAN's learned rain distribution. The inherited SyRaGAN training protocol uses paired clean/rain datasets:

- Rain100L and Rain100H
- Rain800
- Rain1200
- Rain1400
- SPA-data

Consult the original dataset owners and [SyRaGAN repository](https://github.com/jaewoong1/SyRa-Synthesized_Rain_dataset) for access and terms. Do not assume that one dataset's license covers the others.

Each model input is a single side-by-side image:

```text
┌─────────────────────┬─────────────────────┐
│ clean / ground truth│ corresponding rainy│
│      256 × 256      │      256 × 256     │
└─────────────────────┴─────────────────────┘
              combined: 512 × 256
```

The training root requires at least one domain subdirectory because PyTorch `ImageFolder` supplies domain labels. The paper configuration uses two domains and targets domain index `1`.

Before training, run:

```bash
python scripts/validate_data.py data/rains/train --paired
```

The generated `manifest.sha256.json` is the local, exact record of the files used in that run. For a strict reproduction, archive that manifest with the checkpoint and `run_metadata.json`.

## 3. Evaluation data

FID needs a directory of real rainy images and a directory of generated rainy images. LPIPS additionally needs an explicit one-to-one pairing. The evaluator accepts either:

1. identical relative paths in the two roots; or
2. `--pairs-csv pairs.csv`, where the columns are `real,generated` and values are relative to the supplied roots.

Do not report LPIPS from arbitrary sorted files; that makes the score dependent on filenames rather than semantic pairs.

## 4. RF-DETR evaluation

The paper reports RF-DETR mAP on RainGAN-augmented data. The original repository did not include the exact detector configuration, split manifest, checkpoint, or evaluation script. A scientifically complete detector reproduction requires all four. Until they are released, the reported detector values should be labeled “reported in the paper,” not “reproduced from this repository.”
