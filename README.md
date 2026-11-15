# Native 1:1 Co-DeepNet & Marginal-PQ Oracles — Solar Filament Segmentation Challenge 2026

**Author:** Samiul Islam (Islamic University of Technology, Dhaka, Bangladesh — `samiulislam@iut-dhaka.edu`)  
**Technical Report:** [`report/report.pdf`](report/report.pdf) (LaTeX source & vector figures in [`report/`](report/))

This repository contains the complete source code, offline inference notebook, `fp16` ensemble model weights, calibrated meta-scorer coefficients, and reproduced test predictions for our solution to the **Solar Filament Segmentation Challenge 2026** (MAGFiLO GONG H-$\alpha$ dataset).

## Repository Structure

```text
├── requirements.txt                         # Exact Python package versions
├── solar_filament_codeepnet_inference.ipynb # Self-contained end-to-end offline inference pipeline
├── src/                                     # Full modular training, oracle, & evaluation source code
│   ├── physics_v2.py                        # Huber IRLS limb fit & 5-channel native 1888x1888 physics
│   ├── targets_v10.py                       # Inverse-sqrt(area) SDF consensus & singleton ignore shield
│   ├── stage1_detector.py                   # 1408x1408 7-aspect-ratio ignore-shielded Mask R-CNN v2
│   ├── codeepnet_native.py                  # Decoupled Native 1:1 Co-DeepNet (ConvNeXt-S + RibbonBarbUNet)
│   ├── decode_and_eval.py                   # Geodesic hysteresis, Voronoi splits, waist heal, & 24-feat meta-scorer
│   ├── referee_metric.py                    # Exact competition Panoptic Quality (PQ/SQ/RQ) & bootstrap CIs
│   └── run.py                               # Multi-fold end-to-end training & out-of-fold evaluation driver
├── weights/                                 # Trained 3-fold fp16 weights (Git LFS) + meta-scorer JSON
│   ├── stage1_mrcnn_f{0,1,2}_fp16.pt        # Stage-1 Mask R-CNN ResNet-50-FPN v2 fold checkpoints
│   ├── codeepnet_native_f{0,1,2}_fp16.pt    # Stage-2 Native 1:1 Co-DeepNet fold checkpoints
│   └── metascorer_v10.json                  # 5-subfold standardized Ridge + L2 Logistic expected-PQ scorer
├── submissions/
│   └── submission_tier2_multifold_codeepnet_v10.csv # Submitted test predictions (1,183 instances, 0 overlap)
└── report/
    ├── report.pdf                           # Compiled 4-page technical report (+ Acknowledgment & References)
    ├── main.tex, preamble.tex, main.bib     # Official ACM sigconf LaTeX sources
    └── fig1..fig3_*.pdf                     # Vector architecture, quantitative diagnostic, & 1:1 gallery figures
```

## Reproducing Test Predictions Without Retraining

1. Clone the repository with Git LFS enabled so the six `fp16` `.pt` checkpoints in `weights/` are downloaded:
   ```bash
   git lfs install
   git clone https://github.com/samiulabir201/solar-filament-segmentation-challenge-2026.git
   cd solar-filament-segmentation-challenge-2026
   pip install -r requirements.txt
   ```
2. Open and run [`solar_filament_codeepnet_inference.ipynb`](solar_filament_codeepnet_inference.ipynb) from top to bottom (set `TEST_IMG_DIR` to the directory containing the 180 `test_images/*.jpg` files; `WEIGHTS_DIR` defaults automatically to `./weights`).
   - All six checkpoints load strictly offline (`pretrained=False`, zero internet calls required).
   - Writes strictly disjoint compressed COCO RLE predictions matching [`submissions/submission_tier2_multifold_codeepnet_v10.csv`](submissions/submission_tier2_multifold_codeepnet_v10.csv) (`0` intra-image overlapping pixels across all `180` test frames; `53.8 min` total runtime on a single 16 GB NVIDIA Tesla T4).

## Summary of Out-of-Fold & Leaderboard Results

| Evaluation Cohort | Disks | GT Instances | PQ [90% Bootstrap CI] | SQ | RQ | Split / Merge |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled 3-Fold OOF (Folds 0–2)** | **361** | **4,246** | **0.4523 [0.4418, 0.4627]** | **0.6841** | **0.6611** | **113 / 81** |
| Single-Annotator Strata ($A_s = 1$) | 205 | 1,613 | 0.4687 [0.4547, 0.4820] | 0.6896 | 0.6797 | 31 / 33 |
| Dual-Annotator Strata ($A_s = 2$) | 74 | 1,102 | 0.4522 [0.4326, 0.4718] | 0.6844 | 0.6608 | 32 / 19 |
| Triple-Annotator Strata ($A_s = 3$) | 82 | 1,531 | 0.4366 [0.4158, 0.4567] | 0.6784 | 0.6436 | 50 / 29 |
| Zero-Bloat SDF Consensus GT | 361 | 3,081 | 0.4959 [0.4845, 0.5068] | 0.7061 | 0.7023 | 68 / 54 |
| **Pure Multi-Observer ($\ge 2$ Votes)** | **156** | **1,492** | **0.5363 [0.5210, 0.5512]** | **0.7294** | **0.7353** | **29 / 21** |
| **Kaggle Public Leaderboard (Tier-2)** | **180 test** | — | **0.40** (`+0.01` over Tier-1 `0.39`) | — | — | `0` overlaps |
