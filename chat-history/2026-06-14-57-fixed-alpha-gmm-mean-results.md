# Fixed-Alpha GMM Mean Replay Results

**Date**: 2026-06-14

## Context

This session followed up two open questions about the inference-side classifier
evidence:

1. Whether GMM replay using component means, rather than sampled pseudo-features,
   has been evaluated.
2. Whether LR-RGDA+ZS and LADA+ZS have each been evaluated with a tuned ensemble
   coefficient.

The main goal was to avoid an unfair comparison where LR-RGDA+ZS benefits from
an ensemble coefficient while LADA+ZS remains effectively pure LADA.

## Core Conclusion

The strongest current inference-side configuration is:

```text
classifier_feature_transform = test
replay mode = gmm_raw_mean
gmm_fit_space = raw
gmm_sample_mode = mean
num_shots = 16
seeds = 42, 43, 44
num_centers = 4
gmm_k = 4
enable_lada = true
lada_k = 16
rgda_rank = 32
rgda_train_iter = 200
rgda_train_lr = 0.01
lada_train_iter = 200
lada_train_lr = 0.01
fixed LR-RGDA+ZS alpha = 0.05
fixed LADA+ZS alpha = 0.05
ALPHA_SENSITIVITY = 0
```

Under this fixed protocol, not a test-set coefficient sweep, LR-RGDA+ZS clearly
beats LADA+ZS and LADA in all three seeds.

## Result Directories

Remote project root:

```text
/home/raoxuan/projects/project_clip_continual_learning
```

Best fixed-protocol result:

```text
experiments/joint_classifier_alpha005_gmm_mean_fixed
```

Files produced:

```text
manifest.txt
test_gmm_raw_mean_seed42.json
test_gmm_raw_mean_seed43.json
test_gmm_raw_mean_seed44.json
summary.csv
summary.md
```

Macro alpha-sweep sensitivity result, used only to identify and justify the
fixed coefficient:

```text
experiments/joint_classifier_alpha_sweep_gmm_mean_macro
```

Files produced:

```text
alpha_summary.csv
alpha_summary.md
summary.csv
summary.md
test_gmm_raw_mean_seed42.json
test_gmm_raw_mean_seed43.json
test_gmm_raw_mean_seed44.json
```

Older pooled-sample alpha-sweep result, useful only as an oracle/sensitivity
diagnostic, not as the main X-TAIL macro-average evidence:

```text
experiments/joint_classifier_alpha_sweep_gmm_mean
```

## Fixed Alpha = 0.05 Main Result

Remote summary:

```text
experiments/joint_classifier_alpha005_gmm_mean_fixed/summary.md
```

Main fixed-protocol scores:

| Method | Macro Accuracy |
|---|---:|
| CLIP-ZS | 56.40 +/- 0.00 |
| LR-RGDA | 76.93 +/- 0.16 |
| LADA | 76.08 +/- 0.35 |
| LR-RGDA+ZS, fixed alpha=0.05 | 78.44 +/- 0.21 |
| LADA+ZS, fixed alpha=0.05 | 77.03 +/- 0.38 |

Per-seed fixed-alpha scores:

| Seed | LR-RGDA | LADA | LR-RGDA+ZS alpha=0.05 | LADA+ZS alpha=0.05 |
|---:|---:|---:|---:|---:|
| 42 | 76.75 | 75.79 | 78.20 | 76.69 |
| 43 | 76.98 | 75.99 | 78.55 | 76.95 |
| 44 | 77.07 | 76.47 | 78.57 | 77.45 |

Important deltas:

| Comparison | Mean Delta | Min Delta | All Seeds Positive |
|---|---:|---:|---:|
| LR-RGDA - LADA | 0.85 +/- 0.22 | 0.59 | yes |
| LR-RGDA+ZS - LADA | 2.36 +/- 0.24 | 2.09 | yes |
| LR-RGDA+ZS - LADA+ZS | about 1.41 | about 1.12 | yes |

Per-seed LR-RGDA+ZS - LADA:

```text
seed42: +2.41
seed43: +2.57
seed44: +2.09
```

Per-seed LR-RGDA+ZS - LADA+ZS:

```text
seed42: +1.51
seed43: +1.60
seed44: +1.12
```

## Macro Alpha Sweep Result

Remote summary:

```text
experiments/joint_classifier_alpha_sweep_gmm_mean_macro/alpha_summary.md
```

This sweep uses the same dataset-macro average as the main X-TAIL table. It is
still a test-set sensitivity/oracle unless the coefficient is fixed before the
final reporting run.

Best aggregate coefficients:

| Method | Best Alpha | Best Macro Accuracy |
|---|---:|---:|
| LR-RGDA+ZS | 0.05 | 78.44 +/- 0.21 |
| LADA+ZS | 0.05 | 77.03 +/- 0.39 |

Per-seed best coefficients:

| Seed | Best LR alpha | Best LR-RGDA+ZS | Best LADA alpha | Best LADA+ZS |
|---:|---:|---:|---:|---:|
| 42 | 0.05 | 78.20 | 0.05 | 76.69 |
| 43 | 0.05 | 78.55 | 0.05 | 76.95 |
| 44 | 0.05 | 78.56 | 0.05 | 77.45 |

Key alpha grid rows:

| Alpha | LR-RGDA+ZS | LADA+ZS |
|---:|---:|---:|
| 0.00 | 56.40 +/- 0.00 | 56.40 +/- 0.00 |
| 0.05 | 78.44 +/- 0.21 | 77.03 +/- 0.39 |
| 0.10 | 78.14 +/- 0.14 | 76.58 +/- 0.42 |
| 0.50 | 77.12 +/- 0.13 | 76.13 +/- 0.36 |
| 1.00 | 76.93 +/- 0.16 | 76.08 +/- 0.35 |

Interpretation:

```text
alpha=0.05 is consistently best for both LR-RGDA+ZS and LADA+ZS.
LR-RGDA+ZS remains better than LADA+ZS after both methods receive the same
coefficient search space and the same fixed coefficient in the final rerun.
```

## Previous Fixed Alpha Baseline

Before coefficient tuning, the fixed coefficients were:

```text
LR-RGDA+ZS alpha = 0.5
LADA+ZS lada_alpha = 1.0
```

Under that setting:

| Method | Macro Accuracy |
|---|---:|
| CLIP-ZS | 56.40 +/- 0.00 |
| LR-RGDA | 76.93 +/- 0.16 |
| LADA | 76.08 +/- 0.35 |
| LR-RGDA+ZS alpha=0.5 | 77.12 +/- 0.13 |
| LADA+ZS alpha=1.0 | 76.08 +/- 0.35 |

This was already positive for LR-RGDA+ZS, but it was not fully satisfactory
because LADA+ZS with `lada_alpha=1.0` is effectively pure LADA. The fixed
`0.05/0.05` rerun resolves this fairness concern.

## Mean Replay vs Sample Replay

The GMM component-mean replay setting was already the strongest replay variant:

| Replay Setting | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS |
|---|---:|---:|---:|---:|
| GMM raw samples | 76.24 +/- 0.11 | 75.63 +/- 0.28 | 76.46 +/- 0.16 | 75.63 +/- 0.28 |
| GMM sphere samples | 76.22 +/- 0.18 | 75.66 +/- 0.31 | 76.45 +/- 0.17 | 75.66 +/- 0.31 |
| GMM raw component means, alpha=0.5/1.0 | 76.93 +/- 0.16 | 76.08 +/- 0.35 | 77.12 +/- 0.13 | 76.08 +/- 0.35 |
| GMM raw component means, alpha=0.05/0.05 | 76.93 +/- 0.16 | 76.08 +/- 0.35 | 78.44 +/- 0.21 | 77.03 +/- 0.38 |

Conclusion:

```text
Component-mean replay is stronger than stochastic GMM sampling in this setup.
The best current inference-side evidence is gmm_raw_mean + fixed alpha=0.05.
```

## Code Changes

Updated:

```text
main_joint.py
src/utils/main_utils.py
scripts/run_joint_classifier_replay.sh
scripts/summarize_joint_classifier_replay.py
```

Added:

```text
scripts/summarize_joint_alpha_sweep.py
```

Important implementation details:

```text
scripts/run_joint_classifier_replay.sh
  ENSEMBLE_ALPHA controls --alpha.
  LADA_ALPHA controls --lada_alpha.
  Defaults preserve old behavior: ENSEMBLE_ALPHA=0.5, LADA_ALPHA=1.0.

main_joint.py / src/utils/main_utils.py
  alpha sensitivity now supports dataset-macro sweeps matching the main table.
  LADA+ZS receives its own alpha sweep in the same evaluation loop.
```

Verification:

```bash
python -m py_compile main_joint.py src/utils/main_utils.py scripts/summarize_joint_alpha_sweep.py
bash -n scripts/run_joint_classifier_replay.sh
git diff --check -- main_joint.py src/utils/main_utils.py scripts/run_joint_classifier_replay.sh scripts/summarize_joint_classifier_replay.py scripts/summarize_joint_alpha_sweep.py scripts/remote_training_ablation.sh
```

The remote fixed-alpha run completed all three seeds with no Traceback,
OutOfMemoryError, or ERROR in the logs.

## Claim Boundary

Defensible claim after this session:

```text
Under compact GMM component-mean statistical replay, LR-RGDA+ZS with a fixed
low ensemble coefficient (alpha=0.05) outperforms LADA and LADA+ZS under the
same fixed coefficient on all three X-TAIL 16-shot seeds.
```

Still not defensible:

```text
LR-RGDA universally beats LADA.
LR-RGDA beats LADA when real historical 16-shot features are available.
alpha=0.05 is a universally optimal coefficient outside this protocol.
This is an official LADA DPT result.
This is a complete publication-ready result without training-side and
incremental gates.
```

Important boundary retained from earlier experiments:

```text
Real 16-shot features:
  LR-RGDA      77.87 +/- 0.16
  LADA         78.92 +/- 0.16
  LR-RGDA+ZS   78.05 +/- 0.14
  LADA+ZS      78.92 +/- 0.16

Interpretation:
  LADA is stronger when real historical features are available.
  LR-RGDA's current advantage is in compact statistical replay.
```

## Mechanistic Interpretation Update

Current working interpretation:

```text
LR-RGDA + multi-centroid modeling + classifier-side fine-tuning + deterministic
test_transform feature construction + compact GMM component-mean replay + a
smaller ensemble coefficient is the current best-performing inference-side
classifier recipe.
```

The performance gain should not be attributed to a single factor. The current
evidence points to a compound improvement:

1. `classifier_feature_transform=test` improves the feature source used to build
   LR-RGDA and LADA. It avoids random training augmentation noise when extracting
   the 16-shot classifier-construction features.
2. `gmm_sample_mode=mean` is better than stochastic GMM sampling in the current
   protocol. Component means provide compact, low-noise replay points.
3. `num_centers=4` gives LR-RGDA a multi-centroid representation of class
   structure rather than forcing a single Gaussian center per class.
4. `rgda_train_iter=200` fine-tunes LR-RGDA logits/classifier parameters on the
   constructed replay features, improving supervised discrimination.
5. After LR-RGDA classifier fine-tuning, the LR-RGDA logits become more
   confident. A smaller ensemble coefficient is therefore needed when mixing
   with zero-shot CLIP logits. Empirically, `alpha=0.05` is much better than the
   old `alpha=0.5` in the GMM component-mean replay setting.

Important nuance:

```text
The smaller alpha does not mean LR-RGDA is weaker.
It likely means the fine-tuned LR-RGDA logit scale/confidence is stronger, so
only a small LR-RGDA contribution is needed to correct the zero-shot classifier
without overwhelming useful CLIP priors.
```

This interpretation also explains why the same `0.05` coefficient improves
LADA+ZS relative to pure LADA, but LR-RGDA+ZS remains stronger:

```text
Both supervised replay classifiers benefit from a low-weight fusion with CLIP-ZS.
LR-RGDA benefits more under compact GMM component-mean replay, likely because its
regularized Gaussian decision structure and multi-centroid statistics preserve
more useful class geometry than LADA's center-affinity classifier in this
compressed replay regime.
```

## Recommended Next Discussion

Use the following structure for the next detailed review:

1. Separate three evidence levels:
   fixed old alpha, macro oracle sweep, fixed alpha=0.05 rerun.
2. Decide how to present `alpha=0.05`:
   fixed global coefficient, validation-selected coefficient, or sensitivity
   appendix plus fixed-protocol main result.
3. Decide whether the main inference-side table should replace the old
   `LR-RGDA+ZS=77.12` row with the stronger fixed `alpha=0.05` row.
4. Keep the scientific story narrow:
   statistical replay advantage, not universal LADA replacement.
5. Continue only after explicitly deciding whether further no-leak validation
   for alpha selection is required.

## Integration Status: main_joint.py vs main_incremental.py

Checked on 2026-06-14 after the fixed-alpha GMM mean replay result.

Verification command:

```bash
python -m py_compile main_joint.py main_incremental.py src/utils/main_utils.py
```

Result:

```text
compile passed
```

`main_joint.py` has integrated the current best classifier recipe as
configurable options, but the best recipe is not the default. The best observed
recipe must be selected through CLI arguments or the launcher:

```text
--classifier_feature_transform test
--use_gaussian_features
--gmm_fit_space raw
--gmm_sample_mode mean
--gmm_k 4
--num_centers 4
--rgda_train_iter 200
--rgda_train_lr 0.01
--enable_lada
--lada_k 16
--lada_train_iter 200
--lada_train_lr 0.01
--alpha 0.05
--lada_alpha 0.05
```

Current `main_joint.py` defaults remain historical/conservative:

```text
classifier_feature_transform = train
use_gaussian_features = false
gmm_sample_mode = sample
gmm_k = 0
num_centers = 1
rgda_train_iter = 0
enable_lada = false
alpha = 0.5
lada_alpha = 1.0
```

Interpretation:

```text
main_joint.py is ready for the best current inference-side classifier
experiment, as long as the run command explicitly selects the best recipe.
It should not be assumed that simply running main_joint.py with defaults
reproduces the best result.
```

`main_incremental.py` has not yet integrated the current best classifier recipe.
It still follows the older incremental classifier path:

```text
uses build_stats_dict_from_features, not build_multi_center_stats_dict
uses train-transform tr_loader features for classifier construction
does not expose --classifier_feature_transform
does not expose --num_centers
does not expose GMM replay options
does not expose --rgda_train_iter / --rgda_train_lr
does not expose LADA comparison or LADA+ZS alpha options
uses alpha default 0.5
```

Interpretation:

```text
main_incremental.py should not be treated as containing the best current
LR-RGDA + multi-centroid + classifier fine-tuning + test_transform + GMM
component-mean replay recipe. It is either a legacy incremental entry point or
needs a targeted update before it can be used for the current best classifier
claim.
```

Important detail about `test_transform`:

```text
In the best current protocol, `classifier_feature_transform=test` affects the
features used to construct LR-RGDA/LADA and to fit GMM replay. It selects the
deterministic train_loader4updating/test-transform feature path instead of the
randomly augmented train-transform loader. Therefore its benefit is not merely
an evaluation-time transform effect; it changes the classifier-construction and
GMM replay feature source.
```

Practical next step:

```text
Do not change main_joint.py defaults blindly. Prefer a named launcher/preset for
the current best recipe so old baselines remain reproducible. For
main_incremental.py, first decide whether it is still a maintained publication
entry point. If yes, update it deliberately with at least test_transform,
multi-center LR-RGDA, classifier fine-tuning, and alpha=0.05 support; GMM
component-mean replay across incremental history is a larger design change and
should be handled separately.
```

## Best-Recipe Launcher

Added:

```text
scripts/run_best_joint_classifier_recipe.sh
```

Purpose:

```text
Provide a stable, explicit entry point for the current best inference-side
classifier recipe without changing main_joint.py defaults.
```

The wrapper delegates to `scripts/run_joint_classifier_replay.sh` and fixes the
current best protocol through environment defaults:

```text
OUT_DIR = experiments/joint_classifier_best_recipe
REPLAY_MODES = gmm_raw_mean
CLASSIFIER_FEATURE_TRANSFORMS = test
RUN_MEAN_ABLATION = 1
ALPHA_SENSITIVITY = 0
ENSEMBLE_ALPHA = 0.05
LADA_ALPHA = 0.05
```

The underlying launcher already supplies the remaining best-recipe classifier
settings:

```text
--enable_lada
--lada_k 16
--lada_train_iter 200
--lada_train_lr 0.01
--num_centers 4
--rgda_rank 32
--rgda_train_iter 200
--rgda_train_lr 0.01
--gmm_k 4
--gaussian_samples_per_class 16
```

Dry-run verification:

```bash
DRY_RUN=1 OUT_DIR=/private/tmp/joint_best_recipe_dry_run_20260614 \
  SEEDS='42' bash scripts/run_best_joint_classifier_recipe.sh
```

Expanded command confirmed:

```text
python main_joint.py ... --alpha 0.05 --lada_alpha 0.05 --gmm_k 4
--seed 42 --classifier_feature_transform test
--experiment_name test_gmm_raw_mean
--use_gaussian_features --gmm_fit_space raw --gmm_sample_mode mean
```

Interpretation:

```text
This launcher is the preferred reproduction entry for the current best
fixed-protocol inference-side classifier result. It avoids relying on memory or
manual command reconstruction while preserving baseline defaults in
main_joint.py and run_joint_classifier_replay.sh.
```
