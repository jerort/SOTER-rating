# SOTER-rating

Pavement-rating stage of the SOTER pipeline (University of Almería), as reported in
*Field-free automated pavement condition rating from Pleiades Neo imagery via super-resolution
and deep-learning ordinal regression*. The super-resolution stage is in the companion repository
[SOTER-superresolution](https://github.com/jerort/SOTER-superresolution).

A rater is a frozen ImageNet backbone used as a fixed feature extractor, followed by one
trainable linear output and the ordinal thresholds of the BONN loss (a linear probe). Two
architectures (DenseNet-201, ViT-B/16) on three inputs (native, otf, paired) give six candidates,
evaluated by nested cross-validation over five road groups of the 13 labelled roads.

Research code: simple, standalone scripts, kept close to what produced the paper's numbers.

## Names in the code and in the paper

The code keeps the ids of the research runs in file and folder names, so every result traces
back to them. `scripts/rating/nested_cv/export_results.py` writes the results with the paper's names.

| Paper | Code |
|---|---|
| E01, DenseNet-201 | `E04` (`densenet201`) |
| E02, ViT-B/16 | `E11` (`vit_b_16`) |
| input *native* (Pleiades Neo, 0.30 m/px) | `30`, tiles in `rating_30/` |
| input *otf* (enhanced by `otf_soft`, 0.15 m/px) | `15-otf` / `15_otf`, tiles in `rating_15_otf/` |
| input *paired* (enhanced by `paired_soft_2000`, 0.15 m/px) | `15-paired` / `15_paired`, tiles in `rating_15_paired/` |
| test group G1 … G5 | `test_group` 1 … 5, `test1` … `test5`, `refit_t1` … |
| adjacent accuracy | `within1` (`within-1`) |
| AMAE | `mae` of the grade-weighted metrics |
| grade-weighted metrics | `bal_*` in `metrics_all.csv`; all scores in `picks.csv` are grade-weighted |
| training length *E* | `check` (4 checks per epoch) and `epoch` |

## Layout and paper sections

| Paper | Where |
|---|---|
| Road-segment extraction | `scripts/roads/` (centroids and 8 m / 51 m rectangles, cropping), `src/preprocessing/`; geometries in `scripts/roads/{layouts,all_clips,cropping_clips}/` |
| Image labelling | `scripts/rating/data_inspection/` (`labels.xlsx`: consensus label and each specialist's grade; `labels_shp/`) |
| Configuration of experiments | `scripts/rating/nested_cv/head_features.py` (feature cache, head training), `src/rating.py` (backbones, BONN loss and thresholds, batch sampler) |
| Road groups and nested cross-validation | `scripts/rating/nested_cv/common.py` (groups, splits), `nested_cv.py` (searches, refits), split CSVs in `scripts/rating/nested_cv/datasets_pairs/` |
| Metrics and reference points | `scripts/rating/nested_cv/balanced_metrics.py`, `human_reference.py` |
| Results: searches, test scores, training/validation/test QWK | `results/picks.csv`, `results/test_summary.csv`, `results/nested_cv/` |
| Results: two-way ANOVA and Tukey's test | `scripts/rating/nested_cv/anova_letters.py`, `anova_interaction.py` |
| From evaluation to operational use (calibrators, heatmap viewer) | `scripts/rating/calibrate.py`, `merge_predictions.py`, `scripts/visualisation/` (see the note below) |
| Appendix: road-group partition search | `scripts/rating/nested_cv/partition_search.py`, `group_count_sweep.py`, `results/partition_search/` |
| Appendix: settings picked by each search | `results/picks.csv` |
| Figures of the nested cross-validation and its results | `scripts/rating/nested_cv/paper_figures.py` |

## Data

Included: the pavement-condition labels (`scripts/rating/data_inspection/labels.xlsx`), the
road-segment geometries (shapefiles in `scripts/roads/`), and in `scripts/rating/nested_cv/datasets_pairs/`
the 5,354 labelled segments (`pool.csv`) and every split of the nested cross-validation
(`t{t}_v{v}/{train,val,test}.csv` for test group *t* and validation group *v*,
`refit_t{t}/{train,test}.csv` for the refit of outer fold *t*). Each CSV lists
`filepath` (tile path relative to the input's tile folder) and `rating` (consensus grade 1-5).

Not included: the Pleiades Neo and Airbus HD15 imagery are commercial products used under
licence and cannot be shared. The image tiles derived from them may be requested from the
corresponding author, subject to the licence terms and the agreement of SANDO. The backbone
feature caches (16-41 MB each) are rebuilt from the tiles; the ImageNet weights are downloaded
by torchvision and timm.

## Setup

Python 3.11, then `pip install -r requirements.txt` (pinned versions are those the paper's results
were produced with).
Run every command from the repository root with the root on the module path
(Git Bash / Linux shown; in Windows `cmd` use `set PYTHONPATH=.` first):

```bash
export PYTHONPATH=.
export SOTER_DATA_DIR=/path/to/tiles     # holds rating_30/, rating_15_otf/, rating_15_paired/ (default: ./data)
```

## Running the rating pipeline

Steps 1-2 need the image tiles; the others run on the files in this repository.

```bash
# 1. backbone features of every tile, once per rater and input (GPU optional)
python scripts/rating/nested_cv/head_features.py build

# 2. nested cross-validation: 30 searches (2 raters x 3 inputs x 5 test groups) of 50 trials,
#    each followed by its refit and single test; runs on CPU threads, output in
#    scripts/rating/nested_cv/runs/. Safe to interrupt and restart.
python scripts/rating/nested_cv/nested_cv.py run --jobs 6
python scripts/rating/nested_cv/nested_cv.py status
python scripts/rating/nested_cv/nested_cv.py summary          # runs/summary/picks.csv

# 3. results with the paper's names (default: results/nested_cv -> results/; a run of your own:
#    give its folder, the CSVs are written into it)
python scripts/rating/nested_cv/export_results.py [scripts/rating/nested_cv/runs]

# 4. two-way ANOVA (backbone, input, test group as block) and Tukey's test
python scripts/rating/nested_cv/anova_letters.py [picks.csv]
python scripts/rating/nested_cv/anova_interaction.py [picks.csv]

# 5. reference points: human baseline (leave-one-rater-out, ICC) and naive baseline
python scripts/rating/nested_cv/human_reference.py

# 6. road-group partition search (exhaustive, 7,508,501 partitions; a few GB of RAM)
python scripts/rating/nested_cv/partition_search.py
python scripts/rating/nested_cv/group_count_sweep.py

# 7. figures (PDF) into results/figures/
python scripts/rating/nested_cv/paper_figures.py [OUT_DIR] [RESULTS_DIR]
```

The road-segment scripts in `scripts/roads/` and the visualisation scripts take their paths
from variables at the top of each file.

## Results

- `results/picks.csv`: one row per candidate and test group (30 rows), with the paper's names:
  the setting picked by the search (learning rate, weight decay, training length), its
  validation scores (worst and mean of the four validation groups, per group), and the refitted
  model's scores on its four training groups and on the test group.
- `results/test_summary.csv`: per candidate, mean and SD over the five test groups of QWK,
  Spearman, adjacent accuracy, AMAE and RMSE, and the mean training, validation and test QWK.
- `results/human_reference.csv`: the human baseline (ICC, QWK, Spearman, adjacent accuracy,
  AMAE, RMSE by road group) and the naive baseline (`always3_*`).
- `results/partition_search/`: grade counts per road, Pareto fronts of the partition search.
- `results/nested_cv/`: the outputs of `nested_cv.py` with the internal ids. `summary/picks.csv`
  is what steps 3, 4 and 7 read. `refit/<rater>_<input>_test<t>/` holds, for each of the 30
  refits, `head.pt` (the trained linear output and BONN thresholds, a few KB; with the frozen
  ImageNet backbone it is the complete rater), `refit.json` (the picked setting and seed),
  `checks.csv` (training curve), `metrics_all.csv` (natural and grade-weighted metrics on the
  training and test groups) and `preds_test.csv` (per test segment, BONN output `raw` and
  predicted grade `pred`). `curves_E11_15_otf_test3.csv` has the validation curves of the 50
  trials of one search (ViT-B/16, otf, test group G3), shown in the trial figure; the curves and
  Optuna studies of the other searches (about 60 MB) are not included and are regenerated by step 2.

## Calibration and visualisation

`scripts/rating/calibrate.py` fits the Gaussian and ordinal logistic calibrators of
`src/rating.py`, `merge_predictions.py` writes predictions into the labels workbook, and
`scripts/visualisation/` turns them into QGIS layers and the HTML heatmap viewer. These scripts
were written for the earlier full-network regressors (a `best_model.pth` checkpoint whose output
is on the grade scale) and still expect such a checkpoint and their hard-coded paths. They do
not read the linear-probe heads (`head.pt`) of the paper's raters, whose BONN output is turned
into grades by learned thresholds. As the paper says, the calibrators' results are not reported;
the viewer illustrates their use. The earlier training scripts and experiments, which the paper
no longer reports, remain in the git history of this repository.
