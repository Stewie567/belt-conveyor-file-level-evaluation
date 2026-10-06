# Reproducibility details

## Source and experimental population

Upstream dataset: `https://github.com/ArmantasPik/Conveyor-belt-state-classification`, commit `3360d38af8bb614c8f15bb1afe9e742a838f6a8a`.

Files are sorted lexicographically. CSV values are converted to finite three-column float64 arrays and hashed as contiguous little-endian float64 bytes. The first file in each identical numerical-hash group is retained. Label prefixes `[0, 1, 2, 3, 5, 6]` map to class indices `[0, 1, 2, 3, 4, 5]`. File identifiers are parsed into state, rotational speed and repetition. Non-overlapping windows have length and stride 400; incomplete tails are discarded. Numerical values are stored as float32 for training. This produces 509 windows from 78 files.

The first class has 8 files, the next four classes 9 each, and the unloaded damaged class 34. Speed totals are 29/25/24. One numerical duplicate is removed from the 0.5 kg / 540 RPM cell. Full metadata and byte hashes are supplied in `reference/metadata/` and `reference/source_manifest.json`.

## Split experiment

Outer file allocation uses within-class/speed seeded permutations and carries the class fold offset across speeds (`20260908`), preventing unnecessary loss of overall class coverage. The random-window arm uses `StratifiedKFold(3, shuffle=True, random_state=20260908)`. Training seeds are 0, 1 and 2. Both arms fit training-window-only per-channel mean and population standard deviation; discarded tails are excluded. Both use exactly 60 epochs, no early stopping, no augmentation, Adam (0.001 learning rate, 0.0001 weight decay), batches of 64 and six-class weights `N/(6*n_class)`.

Each seed's score is the arithmetic mean of its three outer-fold metrics. Table II reports the mean and sample standard deviation of these three seed scores. It does not pool folds.

The paired gap bootstrap resamples the 78 source files within classes with replacement. Every selected file contributes all its windows, with the same multiplicity applied to both split arms and all three training seeds. Its windows retain their original arm-specific out-of-fold predictions and fold indices. Macro-F1 is recomputed in each fold with all six labels, then averaged over folds and seeds exactly as in Table II. Percentiles 2.5/97.5 form a conditional 95% interval. Fixed seed: `20261007`; replicates: 50,000. The interval is conditional on observed fold assignments and models; it is not an estimate of the pure effect of leakage or a refit interval.

Per-class test-file and test-window counts are output for both arms. A source file may appear in more than one test fold in the window arm, so its fold-specific counts must not be summed as independent acquisitions.

## Exploratory cohort experiment

Seen and unseen arms share target test files and non-target training files. Only seen includes additional target-speed training files. Training sample size, speed composition and selected duration consequently differ. The comparison does not isolate a speed effect at fixed training size.

Both arms standardize using permitted raw training files, including discarded tails. Three inner file folds choose the earliest best validation Macro-F1 epoch, maximum 60 epochs and patience 12. The median selected epoch determines final fitting. Unseen normalization and selection exclude all target-speed files. Nine seen models (three folds × three seeds) are shared across the target-speed comparisons, as in the original study. Original seen selection evidence is saved under `reference/original_seen_selection/`.

File predictions average window softmax probabilities before argmax. For each target speed and seed, predictions are pooled across outer test folds before scoring; seed scores are then averaged. Window-level contrasts use the same pooling convention. Thus the split and cohort experiments use different estimands and should not be combined.

The class-stratified paired target-file bootstrap uses shared file indices for both arms and all seeds. It computes each seed's difference, then averages seeds. The three contrasts use independent deterministic random streams with seeds `20261007 + speed_rpm`, 50,000 resamples each. Unadjusted percentile endpoints are 0.025/0.975. Bonferroni-adjusted endpoints are `0.05/(2*3)` and `1-0.05/(2*3)`: marginal 98.333% intervals with nominal 95% family coverage. Small-sample bootstrap coverage is approximate. All three adjusted intervals include zero.

The originally published draft used 5,000 replicates and seed 20260908 for its unadjusted intervals; this release increases resampling and adds multiplicity adjustment. Point predictions and point estimates are unchanged; percentile endpoints change slightly with the bootstrap run. These intervals exclude refit and session variation.

## Commands and outputs

| Command | Main outputs |
|---|---|
| `python scripts/reconstruct.py --download` | `outputs/reconstruction/table_I.csv`, raw/numerical hashes, metadata, window arrays, split manifests |
| `python scripts/reproduce.py` | `outputs/analysis/table_II.csv`, `table_III.csv`, `test_counts_by_fold.csv`, `split_gap_interval.json`, bootstrap draws, `summary.json` |
| `python -m unittest discover -s tests -v` | Evidence-integrity, grouping and metric-estimand checks |
| `python scripts/train.py --task smoke` | One-epoch synthetic FCN training/forward verification; no paper predictions written |
| `python scripts/train.py --task split` | `outputs/training/MR-01/` |
| `python scripts/train.py --task cohort` | `outputs/training/seen_models/`, `outputs/training/MR-02/` |
| `python scripts/reproduce.py --predictions outputs/training --output outputs/refit-analysis` | Same statistics for newly fitted predictions |

Reference probabilities come from the original complete experimental artifacts. Release packaging sanitizes machine-specific reuse paths in configurations. A SHA-256 manifest verifies the release copies. No model is refitted when reproducing saved-prediction metrics. CPU training is supported but slower; bitwise equality across devices, PyTorch and numerical-library versions is not guaranteed.

## Data and code provenance

Raw signals and original notebooks are obtained from the upstream authors and not redistributed here. This repository does not grant additional rights to the upstream data. The implementation modules and FCN training procedure were extracted from the authors' existing study; reconstruction and analysis entry points were adapted for a standalone release. Unrelated baseline experiments, invalid archived runs, reviewer reports, manuscript files, personal machine paths, model binaries and credentials are excluded.
