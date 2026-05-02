# Robustness Preprocessing Experiments

This experiment checks whether classical image preprocessing changes ValveLens detector behavior under degraded visual conditions. It is separate from the main backend pipeline. The goal is to measure the effect first, then decide whether any preprocessing belongs in runtime inference.

## Why This Exists

ValveLens is meant to work in visually difficult industrial-like scenes. Low light, blur, sensor noise, glare, and weak contrast can reduce detector confidence or cause missed detections. Before changing the live system, this module creates a reproducible benchmark around the existing trained detector.

The experiment does not assume preprocessing always helps. Some filters can recover confidence on one degradation and hurt another. The scripts preserve outputs so the result can be inspected rather than only summarized.

## Data Used

Primary metric source:

- `data/detection/combined/test/images`
- `data/detection/combined/test/labels`

This is the main valve/gauge test set and is the only source used for detector metrics in the first pass.

Synthetic corrupted images are written to:

- `data/robustness/synthetic/low_light/`
- `data/robustness/synthetic/blur/`
- `data/robustness/synthetic/noise/`
- `data/robustness/synthetic/glare/`
- `data/robustness/synthetic/low_contrast/`

Each corruption folder contains `images/` and copied `labels/` so YOLO validation can run.

Qualitative sources:

- existing OpenLORIS folders under `data_sources/extracted/`
- optional ExDARK under `data_sources/extracted/exdark/`

OpenLORIS and ExDARK are not used as valve/gauge detection ground truth here. They are useful for visual examples and discussion of low-light or zone-localization robustness.

## Filters

The preprocessing script supports:

- `none`: copy images unchanged
- `clahe`: CLAHE on the LAB L channel
- `gamma`: lookup-table gamma correction
- `denoise_clahe`: bilateral denoising followed by CLAHE
- `sharpen_clahe`: CLAHE followed by unsharp masking

Diagnostics recorded in manifests:

- variance of Laplacian blur score
- high-intensity glare percentage

## Commands

Prepare folders and check available datasets:

```powershell
python .\scripts\setup_robustness_datasets.py
```

Generate corruptions from the detector test set:

```powershell
python .\scripts\generate_synthetic_corruptions.py --limit 100
```

Run selected preprocessing variants:

```powershell
python .\scripts\preprocess_images.py --source data\robustness\synthetic\low_light --variant clahe --out data\robustness\preprocessed\low_light_clahe
python .\scripts\preprocess_images.py --source data\robustness\synthetic\low_light --variant gamma --out data\robustness\preprocessed\low_light_gamma
python .\scripts\preprocess_images.py --source data\robustness\synthetic\blur --variant sharpen_clahe --out data\robustness\preprocessed\blur_sharpen_clahe
python .\scripts\preprocess_images.py --source data\robustness\synthetic\noise --variant denoise_clahe --out data\robustness\preprocessed\noise_denoise_clahe
python .\scripts\preprocess_images.py --source data\robustness\synthetic\glare --variant clahe --out data\robustness\preprocessed\glare_clahe
```

Create slide-ready previews:

```powershell
python .\scripts\preview_preprocessing_examples.py
```

Evaluate the detector:

```powershell
python .\scripts\evaluate_preprocessing_detector.py
```

The evaluator defaults to `--workers 0` for YOLO validation. Keep that setting on Windows unless there is a reason to use DataLoader workers.

Outputs:

- `artifacts/robustness/synthetic_summary.json`
- `artifacts/robustness/robustness_summary.json`
- `artifacts/robustness/robustness_summary.csv`
- `artifacts/robustness/preprocessing_preview/`

## ExDARK

ExDARK is optional. The setup script does not require it.

If Kaggle CLI is installed and configured:

```powershell
python .\scripts\setup_robustness_datasets.py --download-exdark
```

Manual download option:

```powershell
kaggle datasets download -d washingtongold/exdark-dataset -p data_sources/downloads/exdark --unzip
```

Place extracted files under:

```text
data_sources/extracted/exdark/
```

## Metrics

When labels are available, `evaluate_preprocessing_detector.py` runs YOLO validation and exports:

- precision
- recall
- mAP50
- mAP50-95

It also supports prediction-summary mode for folders without labels:

- image count
- detection count
- mean confidence
- class distribution
- no-detection count

## Limitations

Synthetic corruptions are controlled and repeatable, but they are not a full substitute for field data. They should be used to compare conditions and identify failure modes, not to claim final deployment robustness.

ExDARK is useful for qualitative low-light enhancement tests. It should not be treated as valve/gauge detector ground truth unless its annotations are parsed and mapped to the ValveLens label space later.

The backend inference pipeline is intentionally unchanged by this experiment.
