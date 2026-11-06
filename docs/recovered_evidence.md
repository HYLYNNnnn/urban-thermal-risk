# Recovered segmentation evidence

The supplied `Semantic segmentation.zip` contains the direct upstream results
used in the existing 238-point thermal-risk table. Historical values are retained
with their original definitions. The runnable segmentation implementation in this
repository must be distinguished from these recovered results; rerunning a model
does not automatically reproduce the original experiment.

## Data correspondence

The archive contains 977 perspective images covering 245 point IDs, 242 existing
fisheye images, 977 large three-panel street visualizations and 242 fisheye analysis
visualizations. These are not copied into the repository in bulk. The selected
sample images show the original study's inputs and outputs.

| Recovered CSV | Rows | Correspondence with the 238-point ML table |
| --- | ---: | --- |
| `all_image_results.csv` | 977 | Three per-image metrics aggregate to the saved street-point averages |
| `folder_average_results.csv` | 245 | All 238 final point IDs and all three street metrics match exactly |
| `fisheye_view_factors.csv` | 242 | All 238 final point IDs match; largest factor difference is below 0.00000005, consistent with saved numeric precision |
| `street_canyon_hw_analysis_results.csv` | 245 | All 238 final point IDs and H/W values match exactly |

The audit checks **1,666 final-feature comparisons**: 238 points × seven metrics.
It also checks **735 saved street averages**: 245 points × three metrics, calculated
from the recovered per-image rows. All pass with an absolute tolerance of
`0.000001`. No historical feature values were changed to obtain these matches.

There are 242 points with four perspective images and three points with only
three images: **21, 187 and 242**. All three are absent from the recovered fisheye
table. Across the three point-level sources, 242 IDs have complete recovered
metrics; the final study table uses 238 of them. IDs **9, 172, 178 and 238** have
complete recovered metrics but are absent from the final table. Their exclusion
reason is not preserved, so the audit reports them without adding them to the
historical modeling sample.

The earlier 253-point coordinate input also includes eight IDs absent from the
recovered street batch: **161, 162, 163, 164, 234, 245, 249 and 250**. This is a
comparison of the available files, rather than proof of why those locations were
not retained.

## What the original metrics mean

The historical perspective CSVs use Cityscapes category names and these exact
combinations:

| Historical metric | Class combination |
| --- | --- |
| `green_view_index` | `vegetation + terrain` |
| `sky_visibility` | `sky` |
| `enclosure_index` | `building + wall + fence` |

Point-level values are arithmetic means of the images available for that point.
The label `terrain` is retained here to document the historical calculation;
changing the class set would produce a different definition and a new dataset.

`fisheyes.py` explicitly loads
`nvidia/segformer-b1-finetuned-cityscapes-1024-1024`. It processes an **already
generated** fisheye image; it does not generate that image from perspective views.
Its historical factors are weighted proportions within a circular mask:

- `SVF`: sky;
- `TVF`: vegetation plus terrain;
- `BVF`: building plus wall.

The original code gives pixels a `sin(zenith)` weight, with zenith estimated
linearly from distance to the circle center. The denominator includes all valid
weighted pixels, including other classes; the three reported factors therefore
need not sum to one. The supplied files do not document the projection or camera
calibration sufficiently to treat these values as independently calibrated
physical view factors. The recovered values are preserved as historical study
features, with this limitation recorded.

`hw.py` derives `hw_value` from an empirical combination of building/sky/road
pixel proportions, then averages it per point. This is not a recovered trained
height-to-width network or a surveyed street measurement. Its `confidence` field
is a heuristic based on dispersion, not a calibrated probability.

## Original model and image limitations

Several experimental files have DeepLab, PSPNet or ADE20K in their filenames,
but a filename alone does not establish the loaded model or its category set.
The first four rows in the saved SegFormer test CSV agree with the corresponding
rows in `all_image_results.csv`. The available batch file names and experiments
do not conclusively identify the producer of every historical street result.

The archived `out/` files are plotted **original image / color segmentation /
overlay** panels. They are useful visual evidence but are not lossless numeric
class-ID masks. New inference saves its own masks and provenance separately.

The archive recovers fisheye inputs and analysis code, but a panorama stitching
or perspective-to-fisheye generation program has still not been identified.
Model weights are not included in the supplied archive. The Baidu collection
stage is documented as an upstream image source, while this repository focuses
on segmentation, metric extraction and the existing risk-model analysis.

## Reproduce the read-only audit

Run from the repository root; the audit uses only the Python standard library:

```sh
python -m segmentation.history \
  --data-dir data/historical \
  --features data/processed/historical_features.csv \
  --output outputs/reconciliation.json
```

The JSON report includes input SHA-256 digests, row counts, matching statistics,
extra IDs and explicit issues. Its sibling `reconciliation.csv` contains all
per-point/per-metric comparisons. Missing inputs, conflicting or duplicate IDs,
nonfinite values, and differences outside the tolerance produce a nonzero exit
code. Extra historical IDs are listed rather than silently merged into the
final study. Report paths are prohibited from overwriting any input CSV.

Tests:

```sh
python -m unittest discover -s tests -p test_history.py -v
```
