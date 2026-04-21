# Project 5 - License Plate Deblurring

This repository contains an end-to-end computer vision pipeline for motion deblurring of license plate images.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cleora88/LINA-CV-PROJECT-/blob/main/code/License_Plate_Deblurring.ipynb)

## Repository Structure

- `code/` - Main notebook: `License_Plate_Deblurring.ipynb`
- `data/` - Synthetic and real examples + dataset documentation (`README.txt`)
- `src/` - Reusable Python modules:
  - `synthetic_data.py`
  - `psf_estimation.py`
  - `wiener.py`
  - `metrics.py`
  - `cnn_psf_estimator.py` (optional section)
- `report.pdf` - Final report
- `requirements.txt` - Project dependencies

## Method Summary

1. Generate/prepare blurred license plate data.
2. Estimate motion blur PSF parameters (mainly cepstrum-based).
3. Restore images with Wiener deconvolution.
4. Evaluate with PSNR and SSIM.
5. Compare with an optional CNN PSF estimator.

## Originality Extensions Implemented

The project includes an explicit extension beyond the baseline pipeline:

1. **Changed experimental design**:
   - Three blur/noise regimes (`mild`, `medium`, `hard`)
   - Explicit train/test split for evaluation
2. **Additional restoration baseline**:
   - Added **Richardson-Lucy** comparison against Wiener and inverse filtering
3. **Rewritten narrative and interpretation**:
   - Notebook discussion focused on method behavior, not only final images
4. **Ablation analysis**:
   - Method ablation (Inverse, Wiener fixed-K, Wiener auto-K, Wiener true-PSF, RL)
   - Hyperparameter ablation for Wiener `K` and RL iterations
5. **Updated report**:
   - `report.pdf` summarizes the extended protocol and key findings

## How to Run

1. Create and activate a Python environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run notebook:
   `code/License_Plate_Deblurring.ipynb`

The notebook executes the complete pipeline from data loading/generation to restoration and evaluation.

## Deliverables

- Data folder with inputs and description: `data/`
- One end-to-end notebook: `code/License_Plate_Deblurring.ipynb`
- Report PDF at root: `report.pdf`

## Colab Notes

- The notebook can run directly in Google Colab using the badge above.
- If Colab asks for dependencies, run:
   `!pip install -r requirements.txt`
# LINA-CV-PROJECT-
