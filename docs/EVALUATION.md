# Evaluation

All values below were generated locally on 18 July 2026. The machine used Python 3.12, PyTorch 2.11/CUDA 12.8, and an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM.

## Official NIST MIST phase dataset

The 25 original 16-bit TIFF tiles form a 5×5 acquisition grid. The test exercises high-bit-depth normalization, phase/feature registration, global consistency, exposure balancing, feathering, and coverage-safe cropping.

| Metric | Result |
|---|---:|
| Output geometry | 5,901 × 4,412 |
| Placement RMSE | 0.303 px |
| Placement p95 | 0.584 px |
| Mean seam disagreement | 0.01028 |
| Valid coverage after crop | 1.000 |
| Median confidence | 1.000 |
| Nominal fallbacks | 0 |
| Inconsistent constraints rejected | 0 |
| Local wall-clock runtime | 23.5 s |

The final mosaic was visually inspected at full composition scale: no black gaps, missing interior regions, or tile-block illumination steps were visible.

## Open-world H&E mosaic

`evaluation.open_world_mosaic` samples a 3×3 grid from the held-out OpenSlide H&E region. Each tile receives independent ±5 px stage jitter, per-channel gain in `[0.88, 1.12]`, and bias in `[-0.025, 0.025]`. Ground-truth tile origins remain hidden from the stitcher and are used only for scoring.

| Metric | Hybrid | Learned + gated |
|---|---:|---:|
| Ground-truth placement RMSE | 0.155 px | 0.123 px |
| Ground-truth placement p95 | 0.199 px | 0.156 px |
| Mosaic MAE vs source | 0.02779 | 0.02779 |
| Internal placement p95 | 0.121 px | 0.159 px |
| Seam disagreement | 0.01154 | 0.01154 |
| Coverage | 1.000 | 1.000 |
| Fallbacks | 0 | 0 |
| Quality status | pass | pass |

The learned candidate improved ground-truth placement while the common global optimizer and blender produced the same pixel-rounded mosaic in this test. See `reports/open-world-mosaic.json` for all pair constraints.

## Learned residual gate

RAFT-small was initialized from torchvision's official pretrained weights. Feature and context encoders were frozen; 876,530 update-block parameters were fine-tuned for six epochs on synthetic smooth polynomial distortions applied to source-disjoint NIST patches. The OpenSlide H&E image was never used for optimization.

| Metric | Held-out NIST phase | Held-out H&E |
|---|---:|---:|
| Baseline synthetic-warp MAE | 0.02073 | 0.05220 |
| Gated corrected MAE | 0.02066 | 0.04167 |
| Error reduction | 0.31% | 20.18% |
| Accepted samples | 6.25% | 43.75% |
| Identity MAE | 0.00000 | 0.00000 |

The gate requires no regression on held-out phase images, at least 5% aggregate improvement on held-out H&E, and exact identity behavior. Failed checkpoints are refused by the runtime loader.

`reports/residual-warp-metrics.json` is retained as a negative-control audit record for the first custom U-Net experiment. It is explicitly marked `quality_gate_passed: false`; its checkpoint was removed and the runtime cannot select it.

## Automated checks

The local suite covers configuration boundaries, textured and featureless registration, 16-bit output, single tiles, mixed dimensions, duplicate and missing grid positions, canvas memory limits, intensity-independent crop behavior, ZIP traversal and symlink protection, API job lifecycle, artifact depth, rejected uploads, exact ML identity, warp direction, checkpoint round trips, no-regression gating, and failed-checkpoint refusal.

`python -m evaluation.stress_suite` adds deterministic 5×5 exposure variation, a 1×12 featureless fallback path, a 3×3 16-bit path, and batched learned-gate probes for textured, near-black, flat, and hard-edge inputs. Its machine-readable result is stored in `reports/stress-suite.json`.

## Interpretation limits

These are engineering benchmarks, not clinical performance studies. Synthetic jitter and aberration have known parameters and cannot represent every scanner, objective, stain, tissue preparation, compression, focus failure, or acquisition artifact. There are no cancer labels, so no cancer-detection metric is claimed.
