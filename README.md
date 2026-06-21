# Tile-Based CNN-SVM for UAS Flat-Roof Defect Detection

Code and reproducibility artifacts for the paper *"A Tile-Based CNN-SVM
Architecture for Autonomous UAS Inspection of Flat Roofs"* (Journal of Safety
Science and Resilience, JNLSSR-D-26-00030.R1).

A lightweight tile-based CNN with a linear (squared-hinge) SVM head performs
**binary roof-defect classification** on modified-bitumen flat-roof imagery,
benchmarked against AlexNet and GoogLeNet with seed-averaged 95% confidence
intervals on a **leakage-free, photo-level** train/val/test split.

`Defect = class 0 = positive`. Random seed = 42 throughout.

## Scripts

**Data preparation**
| Script | Purpose |
|---|---|
| `tile_images.py` | Split each full UAS photo into a 6×6 grid of tiles (36 per photo), keeping the 8-digit photo id in every tile name |
| `augment_images.py` | One augmented copy per tile (flip / ±30° rotation / ±50 brightness / Gaussian noise) → `<name>_aug.png` |
| `split_dataset.py` | Leakage-free **photo-level** split (75:15:10, seed 42; augmentations in Training only) → `manifest.csv` |
| `verify_split.py` | Confirms the split is leakage-free (no photo id appears in more than one of train/val/test) |

**Training & evaluation**
| Script | Purpose |
|---|---|
| `run_experiments.py` | Trains CNN-SVM / AlexNet / GoogLeNet × seeds (resumable: ledger + BackupAndRestore); exports metrics, histories, predictions; aggregates mean ± 95% CI |
| `run_ablation.py` | Architecture sweep (2–6 conv × 2–4 dense layers) on the leakage-free split; one process per cell for GPU-memory isolation |
| `run_component_ablation.py` | Leave-one-out ablation of the training recipe (− augmentation, − class weights, Adam vs. SGD), each trained to convergence |
| `run_timing.py` | Inference latency vs. batch size for all three models (random inputs, no dataset needed) → `timing.csv` |

**Analysis**
| Script | Purpose |
|---|---|
| `analyze_predictions.py` | ROC and PR curve data (`roc.dat` / `pr.dat`) + ROC-AUC / AP from a run's `test_predictions.csv` |
| `check_point.py` | Confusion matrix + metrics at the default decision threshold (the paper's operating point) from `test_predictions.csv` |
| `flag_missed_defects.py` | Collects misclassified test tiles into folders (missed defects / false alarms) for error analysis |

Each script reads its input/output paths from constants at the top of the file,
or from the environment variables / command-line arguments documented in its
docstring (e.g. `DATA_DIR`, or a path argument).

## Data, weights & code archives
Openly archived on Zenodo:
- **Dataset** (CC-BY-4.0): DOI [10.5281/zenodo.20594605](https://doi.org/10.5281/zenodo.20594605); includes the reproducible split index `manifest.csv`.
- **Trained model weights**: DOI [10.5281/zenodo.20596485](https://doi.org/10.5281/zenodo.20596485).
- **Code** (archived snapshot of this repository): DOI [10.5281/zenodo.20599364](https://doi.org/10.5281/zenodo.20599364).

## Environment
TensorFlow (GPU) on WSL2. Tested on an NVIDIA RTX 5070 (Blackwell):
```bash
python3 -m venv ~/tf-gpu && source ~/tf-gpu/bin/activate
pip install -r requirements.txt          # installs tensorflow[and-cuda], Pillow, opencv-python, etc.
```

## Reproduce
Each stage is independent; you can start from the published leakage-free split
(Zenodo) and skip stage&nbsp;1. Set the dataset path at the top of each training
script (or via `DATA_DIR`), and run `run_experiments.py` with `SMOKE_TEST=True`
once first to confirm the GPU is used.

```bash
# 1) Data preparation  (or download the published split and skip)
python tile_images.py                 # full photos  -> 6x6 tiles
python augment_images.py              # train-time augmentation
python split_dataset.py               # leakage-free photo-level split -> manifest.csv
python verify_split.py                # confirm no photo leaks across splits

# 2) Train & evaluate
python run_experiments.py             # 3 models x 3 seeds -> metrics + mean ± 95% CI
python run_ablation.py                # architecture sweep         -> ablation_summary.csv
python run_component_ablation.py      # training-recipe ablation   -> component_ablation_summary.csv
python run_timing.py                  # latency vs batch size      -> timing.csv

# 3) Analysis  (point the first two at one run's predictions)
python analyze_predictions.py experiments/cnnsvm_seed42/test_predictions.csv   # -> roc.dat / pr.dat
python check_point.py        experiments/cnnsvm_seed42/test_predictions.csv   # confusion matrix
python flag_missed_defects.py                                                 # collect error tiles
```

## Expected results
Test set (n = 2,540), mean ± 95% CI over three seeds:

| Model | Test accuracy | Recall | Precision |
|---|---|---|---|
| **CNN-SVM (proposed)** | **94.4% ± 0.4%** | 0.920 ± 0.025 | 0.956 ± 0.030 |
| GoogLeNet | 89.2% ± 5.7% | 0.883 ± 0.156 | 0.881 ± 0.040 |
| AlexNet | 79.8% ± 14.0% | 0.672 ± 0.401 | 0.856 ± 0.039 |

## Citation
> S. Dunthorne et al., "A Tile-Based CNN-SVM Architecture for Autonomous UAS
> Inspection of Flat Roofs," Journal of Safety Science and Resilience (under
> revision), 2026.

## License
Code in this repository is released under the **MIT License** (see `LICENSE`).
The dataset is released under **CC-BY-4.0** via the Zenodo record above.
