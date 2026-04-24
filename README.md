# EEG2Motion

[Project Page](#) | [Paper](#) | [Video](#)

## Overview

This repository presents the qualitative results of **EEG2Motion**, an EEG-driven 3D human motion generation framework.

Given EEG signals recorded during a motion-observation paradigm, our method aims to generate semantically consistent and temporally coherent 3D human motion sequences. The results below provide visual comparisons between the ground-truth motion and the EEG-generated prediction.

This page includes:

- A side-by-side motion comparison between **Ground Truth (GT)** and **Prediction (Pred)**.
- Rendered image comparisons for GT and Pred.
- Per-sample qualitative results.
- Additional motion videos for selected samples.

## Qualitative Comparison

### Motion comparison: GT vs. Pred

**Left:** Ground Truth  
**Right:** Prediction

<p align="center">
  <img src="assets/teaser/gt_vs_pred.gif" width="720">
</p>

<p align="center">
  <a href="assets/videos/gt_vs_pred.mp4">Click here to view the original MP4</a>
</p>

## Per-sample Results

Below we show the ground-truth and predicted motion renderings for each sample.

### Sample 1

| Ground Truth (GT) | Prediction (Pred) |
|---|---|
| <img src="assets/samples/sample_01_gt.png" width="360"> | <img src="assets/samples/sample_01_pred.png" width="360"> |

<p align="center">
  <a href="assets/videos/sample_01_gt.mp4">Sample 1 GT Video</a> &nbsp; | &nbsp;
  <a href="assets/videos/sample_01_pred.mp4">Sample 1 Pred Video</a>
</p>

### Sample 2

| Ground Truth (GT) | Prediction (Pred) |
|---|---|
| <img src="assets/samples/sample_02_gt.png" width="360"> | <img src="assets/samples/sample_02_pred.png" width="360"> |

<p align="center">
  <a href="assets/videos/sample_02_gt.mp4">Sample 2 GT Video</a> &nbsp; | &nbsp;
  <a href="assets/videos/sample_02_pred.mp4">Sample 2 Pred Video</a>
</p>

### Sample 3

| Ground Truth (GT) | Prediction (Pred) |
|---|---|
| <img src="assets/samples/sample_03_gt.png" width="360"> | <img src="assets/samples/sample_03_pred.png" width="360"> |

### Sample 4

| Ground Truth (GT) | Prediction (Pred) |
|---|---|
| <img src="assets/samples/sample_04_gt.png" width="360"> | <img src="assets/samples/sample_04_pred.png" width="360"> |

### Sample 5

| Ground Truth (GT) | Prediction (Pred) |
|---|---|
| <img src="assets/samples/sample_05_gt.png" width="360"> | <img src="assets/samples/sample_05_pred.png" width="360"> |

### Sample 6

| Ground Truth (GT) | Prediction (Pred) |
|---|---|
| <img src="assets/samples/sample_06_gt.png" width="360"> | <img src="assets/samples/sample_06_pred.png" width="360"> |

## Repository Structure

```text
EEG2Motion/
├── README.md
├── assets/
│   ├── teaser/
│   │   └── gt_vs_pred.gif
│   ├── samples/
│   │   ├── sample_01_gt.png
│   │   ├── sample_01_pred.png
│   │   ├── sample_02_gt.png
│   │   ├── sample_02_pred.png
│   │   ├── sample_03_gt.png
│   │   ├── sample_03_pred.png
│   │   ├── sample_04_gt.png
│   │   ├── sample_04_pred.png
│   │   ├── sample_05_gt.png
│   │   ├── sample_05_pred.png
│   │   ├── sample_06_gt.png
│   │   └── sample_06_pred.png
│   └── videos/
│       ├── gt_vs_pred.mp4
│       ├── sample_01_gt.mp4
│       ├── sample_01_pred.mp4
│       ├── sample_02_gt.mp4
│       └── sample_02_pred.mp4
