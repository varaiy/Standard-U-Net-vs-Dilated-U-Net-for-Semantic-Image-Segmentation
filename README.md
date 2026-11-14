# Standard U-Net vs Dilated U-Net for Semantic Segmentation

**Controlled single-change study on Pascal VOC 2012: dilation rate 1 vs 2 in the U-Net bottleneck**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Pascal%20VOC%202012-blue)
![Task](https://img.shields.io/badge/Task-Binary%20segmentation-green)
![Status](https://img.shields.io/badge/Status-Preliminary%20(single%20seed)-orange)

Assignment B for **Advanced Computer Vision**, Department of Computer Science and Engineering, National Institute of Technology Raipur (7th semester).

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Method](#method)
- [Experimental Setup](#experimental-setup)
- [Repository Contents](#repository-contents)
- [Reproducing the Experiment](#reproducing-the-experiment)
- [Known Deviations and Limitations](#known-deviations-and-limitations)
- [Future Work](#future-work)
- [References](#references)
- [Author](#author)

---

## Overview

A small U-Net has a theoretical receptive field far smaller than the input image. This project asks:

> Does setting the dilation of the two bottleneck convolutions of a small U-Net to **2** (instead of 1) improve foreground/background segmentation on Pascal VOC 2012, when parameters, data, initialisation, data order and training budget are identical?

Dilated convolutions enlarge the receptive field **without adding parameters**, so the comparison isolates the effect of context size at the bottleneck.

| | Standard U-Net | Dilated U-Net |
|---|---|---|
| Bottleneck dilation | 1 | 2 |
| Theoretical RF at bottleneck output | 68 px | 100 px |
| Trainable parameters | 483,458 | 483,458 |
| Nominal MACs (analytical) | 0.906 G | 0.906 G |

---

## Key Results

Evaluated on the 1,449-image VOC 2012 validation set (used as the held-out test set), 160×160 input, **single run, seed 42**.

| Metric | Standard (d=1) | Dilated (d=2) | Δ | Status |
|---|---:|---:|---:|---|
| Background IoU | 0.7750 | 0.7984 | +0.0235 | measured |
| **Foreground IoU** (20 classes merged) | 0.3100 | 0.4757 | **+0.1657** | measured |
| **mIoU** (2 classes) | 0.5425 | 0.6370 | **+0.0946** | measured |
| Foreground Dice | 0.4732 | 0.6447 | +0.1715 | derived from IoU |
| Trainable parameters | 483,458 | 483,458 | 0 | measured |
| Median latency, batch 1 (ms) | 2.675 | 2.741 | +0.066 (+2.5%) | measured |
| p95 latency (ms) | 2.813 | 3.030 | +0.217 (+7.7%) | measured |
| Best epoch / best val loss | 12 / 0.4397 | 11 / 0.3745 | −0.0652 | measured |

**Takeaways**

- The dilated model has lower validation loss than the baseline at every one of the 12 epochs.
- The mIoU gain comes almost entirely from the foreground class.
- The change costs no extra parameters or nominal MACs, and about 0.07 ms more per image on a Tesla T4.
- **Caveat:** the result rests on one seed and unconverged training (both models were still improving at epoch 12). Treat it as an encouraging preliminary finding, not a statistically established one.

---

## Method

Both models are the same compact `UNetSmall` class. The only difference is a single constructor argument, `bottleneck_dilation`.

**Architecture**

- 3 pooling stages (the original U-Net uses 4), channel widths 16 → 32 → 64 → 128
- Each block: `[3×3 conv → BatchNorm → ReLU] × 2`, size-preserving padding
- Decoder: 2×2 transposed convolutions, concatenation skip connections
- Head: 1×1 convolution to class logits
- Dilated variant: both bottleneck convolutions use `dilation=2, padding=2`

**Theoretical receptive field per convolution output** (recursion `r_l = r_(l-1) + (k_l - 1) * d_l * j_(l-1)`):

```
enc1: 3, 5    enc2: 10, 14    enc3: 24, 32
bottleneck (d=1): 52, 68
bottleneck (d=2): 68, 100
```

On a 160 px input, the bottleneck output covers 42.5% of the image side (standard) vs 62.5% (dilated). These are theoretical upper bounds; the effective receptive field is smaller and was not measured.

---

## Experimental Setup

| Item | Value |
|---|---|
| Dataset | Pascal VOC 2012 segmentation |
| Task formulation | Binary: background (0) vs foreground (20 object classes merged into 1) |
| Ignored label | 255 (void / boundary), masked out of loss and metrics |
| Splits | 1,171 train / 293 validation (80/20 of official train, `random_split`, seed 42) / 1,449 test (official val) |
| Input | 160×160 (image bilinear, mask nearest-neighbour; aspect ratio not preserved) |
| Augmentation | Random horizontal flip, p = 0.5 (train only) |
| Optimiser | Adam, lr = 1e-3, constant, no scheduler |
| Loss | Cross-entropy, `ignore_index=255`, no class weights |
| Batch size / epochs | 8 / 12 (fixed) |
| Checkpointing | Lowest validation loss |
| Determinism | Seeds for `random`, `numpy`, `torch`, CUDA; `cudnn.deterministic=True`, `benchmark=False` |
| Hardware | Tesla T4 (Google Colab) |
| Software | PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130 |

**Latency protocol:** forward pass only, batch size 1, one test image, 10 warm-up + 50 timed runs, `torch.cuda.synchronize()` after each run, FP32, `torch.inference_mode`.

---

## Repository Contents

| File | Description |
|---|---|
| `Pet_Segmentation_Experiment.ipynb` | Project notebook (training, evaluation, latency, figures) |
| `Standard_vs_Dilated_UNet_VOC2012_Report.pdf` | Full assignment report |
| `modified_unet_best.pth` | Best checkpoint of the dilated U-Net |
| `modified_history.json` | Per-epoch train/val loss of the dilated U-Net |
| `evaluation_results.json` | IoU, parameters, latency, software versions |
| `model_comparison.csv` | Side-by-side metric table |
| `segmentation_comparison.png` | Qualitative figure (failure cases only) |

> The baseline checkpoint and `baseline_history.json` were not preserved; baseline epoch losses come from the notebook log (4 decimals).

---

## Reproducing the Experiment

```bash
# 1. Install dependencies
pip install torch torchvision numpy matplotlib

# 2. Open the notebook (a GPU runtime such as Colab T4 is recommended)
jupyter notebook Pet_Segmentation_Experiment.ipynb
```

Run the notebook top to bottom. It downloads VOC 2012, builds the split with seed 42, trains both models (`bottleneck_dilation=1` and `bottleneck_dilation=2`), and runs the evaluation and latency steps.

---

## Known Deviations and Limitations

**Deviations from the project brief**

- The brief specifies **21 output classes**; the executed experiment merged the 20 object classes into one foreground class (**2 output classes**). Per-class 20-class IoU does not exist.
- Dice is derived exactly from IoU. FLOPs/MACs are analytical (not profiler-measured). Precision, recall, pixel accuracy and confusion matrices were **not saved** and are marked NOT MEASURED.
- Notebook and result files use the words "Pet" / "Oxford-IIIT Pet" (leftover from an earlier template). The data is VOC 2012, and `evaluation_results.json` contains a hard-coded `"dataset": "Oxford-IIIT Pet"` string that should be corrected.
- No published standard-vs-dilated U-Net comparison on VOC 2012 under matched conditions was found, so published values are marked NOT REPORTED.

**Limitations**

- Single seed, no confidence intervals or significance test.
- Short fixed schedule (12 epochs); neither model converged.
- Small 0.48 M-parameter network, no pretraining, 1,464 training images (SBD not used), low 160×160 resolution.
- Latency measured on one image on one GPU.
- Possible gridding artefacts from two consecutive d = 2 convolutions at 20×20 resolution (not tested).
- Qualitative figure contains only failure cases (small, distant or low-contrast objects), biased toward baseline failures.
- Results are on VOC val used as test, so they are not comparable to leaderboard numbers.

---

## Future Work

1. Repeat with at least five seeds; report mean ± std and a paired bootstrap over test images.
2. Train longer with an LR schedule and early stopping so both models converge.
3. Save confusion matrices; report pixel accuracy, precision, recall and Dice directly.
4. Ablate dilation rates (1, 2, 3, 4), mixed rates (e.g. 1, 2, 3) to avoid gridding, and dilation in other blocks.
5. Measure the effective receptive field empirically and relate it to the IoU gain.
6. Compare against other context mechanisms (U-Net++, attention gates) at matched parameter budgets.
7. Move to 21-class VOC with per-class IoU, higher resolution, SBD data and a pretrained encoder.

---

## References

1. Everingham et al., "The Pascal Visual Object Classes (VOC) Challenge," *IJCV*, 2010.
2. Long, Shelhamer, Darrell, "Fully Convolutional Networks for Semantic Segmentation," *CVPR*, 2015.
3. Ronneberger, Fischer, Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *MICCAI*, 2015.
4. Yu, Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions," *ICLR*, 2016.
5. Chen et al., "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs," *IEEE TPAMI*, 2018.
6. Luo, Li, Urtasun, Zemel, "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks," *NeurIPS*, 2016.
7. Wang et al., "Understanding Convolution for Semantic Segmentation," *WACV*, 2018.
8. Zhou et al., "UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation," *IEEE TMI*, 2020.

The full reference list is in the report PDF.

---

## Author

**Kumar Varaiy**
Roll No. 23115051 · Enrolment No. 230342
B.Tech Computer Science and Engineering, 7th Semester
National Institute of Technology Raipur
