# Urban Thermal Risk Analysis

A Python research workflow connecting street-scene semantic segmentation with environmental indicators and an interpretable XGBoost model of a constructed urban thermal-risk index (THIR).

The project focuses on **PyTorch segmentation, auditable pixel metrics, point-level data integration and XGBoost/SHAP**. Street imagery was obtained upstream using Baidu Maps according to the study's data collection workflow. This repository takes local images as input; no Baidu API key is required.

```mermaid
flowchart LR
    A[Local street and fisheye images] --> B[PyTorch SegFormer Cityscapes]
    B --> C[Class masks and image metrics]
    C --> E[Join by point ID]
    D[Existing GIS and context features] --> E
    E --> F[HEVA and THIR index]
    F --> G[XGBoost and SHAP]
```

## What is included

- A runnable **SegFormer-B1** inference entry with pinned NVIDIA checkpoint revisions, strict Cityscapes 19-class validation, CPU/CUDA/MPS selection and optional B5.
- Raw numeric label masks, color masks, overlays, per-image metrics, per-point averages and a model/run manifest.
- Street GVI/sky/enclosure metrics and historical weighted fisheye SVF/TVF/BVF proxies; vegetation-only fractions are also exposed.
- An explicit join with existing GIS/context data, with point identity, coordinate, completeness and metric-policy checks.
- The recovered historical CSVs and 238-point study table, plus a read-only reconciliation tool.
- A repaired THIR/XGBoost/SHAP workflow with independent train/validation/test partitions and validation-only early stopping.
- Five original sample images and curated outputs. The 4.2 GB archive, full image batches, model weights and local environments are not included.

## Main workflow and entry points

The main workflow starts with local street/fisheye images, extracts environmental indicators, joins them to the study's existing GIS/context data, and models the constructed THIR index with XGBoost and SHAP. The commands below run from this repository's root.

| Entry point | Purpose |
| --- | --- |
| [`segmentation/infer.py`](segmentation/infer.py) | Segment street or existing fisheye images and export masks, visualizations and indicators |
| [`segmentation/merge.py`](segmentation/merge.py) | Join the new image indicators to context features by point ID |
| [`segmentation/history.py`](segmentation/history.py) | Check recovered historical indicators against the original study table |
| [`analysis/train.py`](analysis/train.py) | Calculate HEVA/THIR, train/evaluate XGBoost and export SHAP explanations |

### About the original main.py

[`reference/original_analysis/main.py`](reference/original_analysis/main.py) is the original analysis script. It loads the study CSV, calculates HEVA/THIR, trains XGBoost, generates SHAP plots, assesses renovation needs and exports result tables and spatial plots. [`explain.py`](reference/original_analysis/explain.py) contains the more detailed training diagnostics and SHAP visualizations. These snapshots document the original research and the figures listed in the historical gallery below.

The supported analysis command is `python -m analysis.train ...`, shown in step 4. It includes the repaired risk classification and separate training/validation/test evaluation. The historical scripts and the supported modules have different output names; the gallery records original figures separately from the current example outputs.

## Publish this folder

Add the entire **`urban-thermal-risk`** folder to GitHub Desktop as one repository. It contains the source, dependency lists, small sample inputs and documentation. Its current upload contents are approximately 2.6 MB. Model weights, local environments and generated experiment folders are excluded by `.gitignore`. The original 4.2 GB archive remains outside this repository.

Original SHAP figures can be added later under [`docs/historical-figures/`](docs/historical-figures/README.md). The reserved titles and image lines below are ready for that step.

## Quick start

Use Python 3.12. From this repository's root:

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Then install the verified direct dependencies:

```sh
python -m pip install -r requirements.txt
```

On macOS, XGBoost requires an OpenMP runtime. If it reports a missing `libomp.dylib`, install `libomp` through your local package manager; this follows the [official XGBoost installation guidance](https://xgboost.readthedocs.io/en/stable/install.html). The environment used for verification is recorded in `requirements-lock.macos-arm64.txt`; GPU-specific installations can follow the [official PyTorch selector](https://pytorch.org/get-started/locally/).

## 1. Segment the supplied samples

```sh
python -m segmentation.infer --input data/samples/street --output outputs/street --view street
python -m segmentation.infer --input data/samples/fisheye --output outputs/fisheye --view fisheye
```

The first inference downloads the official B1 checkpoint (approximately 55 MB) to the normal Hugging Face cache. Later runs can use the cached files. The checkpoint is not committed to Git. The default is CPU; use `--device cuda` or `--device mps` only on compatible hardware. `--max-edge` limits input size while keeping its aspect ratio; lowering it may change predictions.

Inputs follow the recovered naming convention:

```text
street/1/1_121.620895_38.918283_0.png
street/1/1_121.620895_38.918283_90.png
street/1/1_121.620895_38.918283_180.png
street/1/1_121.620895_38.918283_270.png
fisheye/1/1_fisheye.jpg
```

Outputs are `raw_masks/`, `color_masks/`, `overlays/`, `metrics.csv`, `point_metrics.csv` and `run.json`. Raw masks contain Cityscapes IDs 0–18; 255 marks ignored pixels outside the fisheye circle. Failed images or initialization produce an explicit failed manifest and nonzero exit. Choose a new output directory for each experiment.

The implementation uses the [NVIDIA SegFormer-B1 Cityscapes checkpoint](https://huggingface.co/nvidia/segformer-b1-finetuned-cityscapes-1024-1024) through [Transformers' SegFormer implementation](https://huggingface.co/docs/transformers/model_doc/segformer). Several recovered filenames say DeepLab although they load SegFormer; the formal entry is named for its actual architecture.

## 2. Join sample metrics with existing context

```sh
python -m segmentation.merge --street outputs/street/point_metrics.csv --fisheye outputs/fisheye/point_metrics.csv --context data/processed/historical_features.csv --output outputs/sample-features.csv --allow-partial
```

Only point 1 has supplied sample images. This command explicitly writes one merged point and reports the other 237 context points as omitted. It does not copy their old image metrics into a supposedly new inference result. Without `--allow-partial`, missing context points cause failure. Four distinct cardinal street views and one fisheye result are required per merged point.

The existing context supplies H/W, housing prices, population, POI and road density. These are not reconstructed by the segmentation model. The join replaces the six image-derived columns and preserves the independent context columns.

## 3. Verify recovered historical results

```sh
python -m segmentation.history --data-dir data/historical --features data/processed/historical_features.csv --output outputs/reconciliation.json
```

The recovered upstream tables match all **1,666 historical image/HW feature values** in the final 238-point table. Another 735 comparisons verify street-image averages. The audit retains extra historical IDs and their provenance rather than silently expanding the final sample. Details: [Recovered evidence](docs/recovered_evidence.md).

## 4. Run XGBoost and SHAP on the historical study

```sh
python -m analysis.train --input data/processed/historical_features.csv --output outputs/analysis --max-estimators 100 --early-stopping-rounds 15
```

The one-point sample is for checking segmentation/join behavior, not model evaluation. Use the full historical study table for this command. It exports the THIR/risk table, split IDs, model JSON, test metrics and SHAP plots/values. Early stopping uses only validation rows; training-only five-fold CV does not inspect the test partition.

THIR is **constructed from the environmental features with fixed HEVA weights**. This model fits that constructed index, not an independently measured temperature target. Random point splits do not establish spatial generalization, and SHAP does not establish causal cooling effects.

## Original study figures

This gallery is reserved for figures generated during the original study. The figure titles and filenames below identify where each archived image belongs. Current verification outputs are kept in `docs/examples/`.

To add a figure, copy it to `docs/historical-figures/` using the indicated filename, then remove the `<!--` and `-->` around its prepared image line in this README. The [Chinese figure guide](docs/historical-figures/README.md) lists equivalent filenames from the earlier analysis version.

### SHAP feature importance

Mean absolute SHAP values summarize each feature's average contribution to the model's THIR prediction.

Original figure pending: `shap_importance_explained.png`.

<!-- ![Original study SHAP feature importance](docs/historical-figures/shap_importance_explained.png) -->

### SHAP summary plot

The beeswarm plot shows the direction and distribution of feature contributions across study points.

Original figure pending: `shap_summary_explained.png`.

<!-- ![Original study SHAP summary plot](docs/historical-figures/shap_summary_explained.png) -->

### SHAP dependence plots

Feature values are plotted against their SHAP contributions, including green view index and tree/sky view factors.

Original figure pending: `shap_dependence_detailed.png`.

<!-- ![Original study SHAP dependence plots](docs/historical-figures/shap_dependence_detailed.png) -->

### Green view index dependence

Reserved for the original individual green view index dependence plot.

Original figure pending: `shap_dependence_gvi_english.png`.

<!-- ![Original study green view index SHAP dependence](docs/historical-figures/shap_dependence_gvi_english.png) -->

### Tree view factor dependence

Reserved for the original individual tree view factor dependence plot.

Original figure pending: `shap_dependence_tvf_english.png`.

<!-- ![Original study tree view factor SHAP dependence](docs/historical-figures/shap_dependence_tvf_english.png) -->

### Green view index and tree view factor interaction

Reserved for the original dependence plot colored by the second feature; this visualization is distinct from a computed SHAP interaction-value matrix.

Original figure pending: `shap_interaction_explained.png`.

<!-- ![Original study green view index and tree view factor interaction](docs/historical-figures/shap_interaction_explained.png) -->

### Individual point explanation

Force and waterfall plots illustrate the feature contributions for a selected study point.

Original figure pending: `shap_individual_explained.png`.

<!-- ![Original study individual point SHAP explanation](docs/historical-figures/shap_individual_explained.png) -->

### XGBoost training and evaluation

Reserved for the original training diagnostics, predicted-versus-target plot and residual distribution. Any scores shown here describe that historical experiment and its original evaluation procedure.

Original figure pending: `xgboost_detailed_analysis.png`.

<!-- ![Original study XGBoost training and evaluation](docs/historical-figures/xgboost_detailed_analysis.png) -->

### Renovation priority analysis

Reserved for the original feature-based renovation priority matrix.

Original figure pending: `renovation_priority_matrix.png`.

<!-- ![Original study renovation priority matrix](docs/historical-figures/renovation_priority_matrix.png) -->

## Metric definitions

The default preserves the historical study's definition:

| Metric | Classes and support |
| --- | --- |
| Street `green_view_index` | vegetation + terrain / all image pixels |
| Street `sky_visibility` | sky / all image pixels |
| Street `enclosure_index` | building + wall + fence / all image pixels |
| Fisheye `SVF` | sky / all weighted circle pixels |
| Fisheye `TVF` | vegetation + terrain / all weighted circle pixels |
| Fisheye `BVF` | building + wall / all weighted circle pixels |

`terrain` is not synonymous with plants. Both street and fisheye exports therefore include a separate `vegetation_only_fraction`. The historical fisheye calculation gives each circle pixel a `sin(theta)` weight based on radial distance; projection/calibration information is insufficient to call these independently calibrated physical radiation view factors. Other classes remain in the denominator, so SVF+TVF+BVF need not equal one.

New inference is recorded separately from the original results: the default street model size and aspect-preserving preprocessing differ from some archived experiments. Successfully running the code does not establish segmentation accuracy; no hand-labeled validation masks or mIoU benchmark were supplied.

## Verification

```sh
python -m unittest discover -s tests -v
python -m unittest discover -s analysis/tests -t . -v
```

The checks cover label mismatch, pixel/weighted metrics, image identity, point aggregation, failed batches, data joins, historical correspondence, risk boundaries, split isolation and real XGBoost/SHAP output consistency. CPU inference was actually run on four street images and one fisheye image, followed by a one-point join. The 238-point historical analysis was also executed. CUDA/MPS and the full 1,219-image batch were not rerun.

Curated real outputs:

![Street segmentation overlay](docs/examples/street-overlay.png)

![Fisheye segmentation overlay](docs/examples/fisheye-overlay.png)

## Source layout and limits

- `segmentation/`: supported inference, metrics, join and historical audit.
- `analysis/`: supported THIR model and explanations.
- `data/samples/`: five original study images.
- `data/historical/`: recovered intermediate CSVs, retained unchanged.
- `data/processed/`: the original 238-point feature table.
- `reference/`: unchanged historical experiments; see its README before reading/running them.
- `docs/`: provenance, curated outputs and Chinese instructions.

Baidu acquisition is a summarized upstream step. Panorama stitching/perspective-to-fisheye generation, complete GIS engineering and a trained H/W network were not recovered. Existing fisheye inputs and empirical H/W results are supplied; a panorama projection or new GIS layers are not generated by this repository.

Chinese guide: [使用说明](docs/使用说明.zh-CN.md). Repair and validation notes: [修复说明](docs/修复说明.zh-CN.md).
