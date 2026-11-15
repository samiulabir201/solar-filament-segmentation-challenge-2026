# Solar Filament Segmentation Challenge 2026 — Reproducible Inference Pipeline (Paradigm v10)

## Pipeline Overview
1. **Physics-Informed 5-Channel Native 1:1 Representation (`physics_v2`)**:
   - Fits exact sub-pixel solar disk center `(cx, cy)` and radius `R_sun` via quantile bright-limb circle fit.
   - Applies radial limb-darkening flattening `I_flat`, bright-plage-clamped multi-scale dark depth `D_multi` across scales `[7, 15, 29, 55]`, and 8-orientation anisotropic ribbonness `T_ribbon` inside the native `1888x1888` active-disk slice (`1:1` integer pixel coordinates, zero interpolation blur), plus on-GPU local `15x15` fine barb contrast and `mu_disk = sqrt(max(0, 1 - rho^2))`.
2. **Stage-1 Complementary 1408px Mask R-CNN v2 Detector**:
   - ResNet50-FPN v2 with filament-calibrated anchor ratios `(0.15..6.5)` and `28x28` RoIAlign mask heads.
   - Multi-fold `4x` flip TTA proposals clustered across all folds/views (`iou_thr=0.45`) for object-level touch separation and false-positive rejection.
3. **Heterogeneous Cooperative 1:1 Co-DeepNet (Peer A ConvNeXt-Small + Peer B StripBarbUNet)**:
   - Bidirectional zero-initialized `CrossBridge` adapters across 4 spatial scales (`1/16, 1/8, 1/4, 1/2`), Peer A native boundary stem + ImageNet-22k ConvNeXt-Small encoder, and Peer B stride-1 anisotropic `(1x9, 9x1, 3x3, dilated-3x3)` `RibbonBarbBlock` decoder.
   - Trained with mutual Co-Teaching on annotator-noisy regions (`A_s = 1`), bit-4 thin-filament `2.0x` focal boosting, and `3`-fold ensemble full-disk `1888x1888` `4x` flip TTA.
4. **Geodesic Hysteresis + 24-Feature Expected-PQ Meta-Scorer**:
   - Seed-to-low-threshold `binary_propagation` hysteresis decoding, detector-verified touch separation, waist-only collinear fragment healing, and 5-subfold chronological Ridge+Logistic expected-PQ scoring over 24 orthogonal features (`metascorer_v10.json`).

## Reproducing Predictions Offline (No Internet Required)
1. Place the `fp16` checkpoint weights (`stage1_mrcnn_f{0,1,2}_fp16.pt`, `codeepnet_native_f{0,1,2}_fp16.pt`) and `metascorer_v10.json` under the weights directory (`WEIGHTS_DIR`, or attached Kaggle Dataset).
2. Run `deliverables/kaggle_inference_kernel/solar_filament_codeepnet_inference.ipynb` with `TEST_IMG_DIR` pointing to `MAGFiLO_1.0_Kaggle_2026/test/test_images`. All models load with `pretrained=False` and `enable_internet=false`.
3. Output `submission.csv` (`filament_id,segmentation_rle` with compressed COCO RLE counts, strict zero intra-image pixel overlap).

## Verified Results & Kaggle Reproducibility
### Node 29 (`029_prod_multifold_hybrid_codeepnet`) Out-of-Fold Validation
- **3-Fold Pooled OOF (`361` stems, folds `0, 1, 2`)**: `PQ = 0.4523` (`0.452273`, 90% CI `[0.4418, 0.4627]`, `SQ = 0.6841`, `RQ = 0.6611`, `TP = 2687`, `FP = 1196`, `FN = 1559`, `split = 113`, `merge = 81`).
- **Fold-0 Paired Comparison (`119` stems)**: `PQ = 0.4473` vs Node 23 single-fold `0.4457` (`gain = +0.0016`, `p_better = 0.6635`).
- **Clean Consensus Subsets**: `consensus_pq = 0.4959`, `pure_votes2_consensus_pq = 0.5363` (`SQ = 0.7294`, `RQ = 0.7353`), `As1 = 0.4687 / As2 = 0.4522 / As3 = 0.4366`.
- **Integrity Gates**: `vault_read_count = 0`, `stems_with_overlap_val = 0`, `stems_with_overlap_test = 0`, `competition.check_submission == []`.

### Kaggle Offline Deployment & Public Leaderboard (`filament-segmentation-2026`)
- **Private Weights Dataset**: `samiulislam180041221/solar-filament-codeepnet-2026-weights` (`6` `fp16` `.pt` checkpoints + `metascorer_v10.json`).
- **Offline GPU Inference Notebook**: `samiulislam180041221/solar-filament-codeepnet-2026-inference` (`enable_internet: false`, `pretrained=False`; completed `180/180` test images on Kaggle Tesla T4 `14.6 GB` in `53.82m`, `1183` rows across `177/180` stems, `overlap_test = 0`, `mean_union_iou = 0.984601` parity vs Colab A100).
- **Tier-1 Submission (`ref=58351059`)**: `deliverables/submissions/submission_tier1_fold0_5model_consensus.csv` (`1204` rows, `177/180` stems) — **Public LB = `0.39`** (`SubmissionStatus.COMPLETE`).
- **Tier-2 Submission (`ref=58359160`)**: `deliverables/submissions/submission_tier2_multifold_codeepnet_v10.csv` (`1183` rows, `177/180` stems) — **Public LB = `0.40`** (`SubmissionStatus.COMPLETE`).
