# Tile-Based CNN-SVM for UAS Flat-Roof Defect Detection

Code and reproducibility artifacts for the paper *"A Tile-Based CNN-SVM
Architecture for Autonomous UAS Inspection of Flat Roofs"* (Journal of Safety
Science and Resilience, JNLSSR-D-26-00030.R1).

A lightweight tile-based CNN with a linear (squared-hinge) SVM head performs
**binary roof-defect classification** on modified-bitumen flat-roof imagery,
benchmarked against AlexNet and GoogLeNet with seed-averaged 95% confidence
intervals on a **leakage-free, photo-level** train/val/test split.

## Pipeline
| Script | Purpose |
|---|---|
| `tile_images.py` | Split full UAS photos into an N×N grid of tiles (6×6 = 36), preserving the 8-digit photo id in every tile name |
| `augment_images.py` | One augmented copy per tile (flip / ±30° rotation / ±50 brightness / Gaussian noise) → `<name>_aug.png` |
| `resplit_dataset.py` | Leakage-free **photo-level** re-split (75:15:10, seed 42; augmentations in Training only) → `manifest.csv` |
| `run_experiments.py` | Trains CNN-SVM / AlexNet / GoogLeNet × seeds; resumable (BackupAndRestore + ledger); exports metrics, histories, predictions; aggregates mean ± 95% CI |
| `analyze_predictions.py` | ROC/PR curves (`roc.dat`/`pr.dat`) + ROC-AUC / AP from a run's `test_predictions.csv` |
| `check_point.py` | Selects the recall≈0.95 operating point for the confusion matrix |

Defect = class 0 = positive. Random seed = 42 throughout.

## Data & trained weights
Archived on Zenodo (CC-BY-4.0): **DOI: 10.5281/zenodo.20594605** (dataset + weights).
The reproducible split index `manifest.csv` is included in that record.

## Environment
TensorFlow (GPU) on WSL2 — see `requirements.txt`. Tested on an NVIDIA RTX 5070
(Blackwell): WSL2 + `pip install "tensorflow[and-cuda]"`.

## Reproduce
```bash
# 1. download the dataset from Zenodo; set SRC/DST/DATA paths in the scripts
python resplit_dataset.py     # build the leakage-free split (or use the published split)
python run_experiments.py     # SMOKE_TEST=True to verify, then False for the full 9 runs
```

## Citation
> S. Dunthorne et al., "A Tile-Based CNN-SVM Architecture for Autonomous UAS
> Inspection of Flat Roofs," Journal of Safety Science and Resilience (under
> revision), 2026.

## License
See `LICENSE` (MIT).
