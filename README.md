# Belt conveyor file-level evaluation

Reproducibility code for **Recording-Disjoint Evaluation of Belt Conveyor State Monitoring Across Operating Speeds**, IEEE SIBIRCON 2026, by Jianfei Wang, Sergey A. Gordin, and Kirill A. Varnavskiy.

The paper compares random window splits with file-disjoint splits of the same public strain-related signal dataset. A common FCN with 400-sample inputs and a fixed 60-epoch budget obtains window-level Macro-F1 of **55.84% versus 29.50%**, a **26.34 percentage-point gap**. The second experiment is an **exploratory training-cohort comparison** on identical held-out target files; training size, speed mixture and selected epoch count differ between its arms. Its three multiplicity-adjusted intervals include zero.

## What is included

- Portable reconstruction, numerical hashing and fold-allocation code: **79 source CSV files → 78 distinct numerical files → 509 non-overlapping windows**.
- Full test probabilities and configurations for the **18 split-comparison units and 54 cohort-comparison units**, with SHA-256 evidence hashes.
- Source-only inner model-selection records and original environment information.
- Scripts reproducing all numerical paper tables, per-class fold counts, the paired split-gap interval and multiplicity-adjusted cohort intervals.
- A training entry point implementing the two paper experiments, without dependencies on previous project logs or baseline studies.

Raw signals, the accepted manuscript, reviewer reports, credentials, virtual environments and model checkpoints are not included. Obtain raw signals from the original dataset authors using the command below.

## Quick numerical reproduction

Use Python 3.12. From this repository directory:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
python scripts/reproduce.py
python -m unittest discover -s tests -v
```

This verifies the release evidence and writes CSV tables and statistical details to `outputs/analysis/`. No raw signals or GPU are needed to reproduce reported test metrics and conditional intervals from the saved predictions.

## Reconstruct the dataset

```bash
python scripts/reconstruct.py --download
```

The script obtains only the 79 source CSV files from [the original public repository](https://github.com/ArmantasPik/Conveyor-belt-state-classification), pinned to commit `3360d38af8bb614c8f15bb1afe9e742a838f6a8a`. Every byte hash is checked. Original data are credited to their source authors; downloading from them does not imply a new license from this repository.

Alternatively, use an existing original dataset directory:

```bash
python scripts/reconstruct.py --data-dir /path/to/RawData
```

Outputs are `table_I.csv`, metadata, duplicate identification, `windows_400.npz`, window indices and the exact outer/inner file-fold manifests. The arrays and fold assignments are checked against the release reference metadata.

## Retrain the FCN experiments

```bash
python -m pip install -r requirements.txt
python scripts/train.py --task smoke
python scripts/train.py --task split
python scripts/train.py --task cohort
```

For a local original data directory, pass `--data-dir /path/to/RawData` to the training command as well. Run reconstruction first. Training outputs are written separately to `outputs/training/`; saved reference predictions are preserved. The cohort command fits one seen model per fold and seed and reuses it across target speeds, matching the original experiment. It independently selects the duration for unseen cohorts using three inner file folds, at most 60 epochs and patience 12.

`python scripts/reproduce.py --predictions outputs/training --output outputs/refit-analysis` analyzes a completed retraining run. Numerical equality across hardware and PyTorch versions is not promised. `docs/original_environment.md` and `docs/requirements-original.txt` record the original Windows/CUDA environment; the latter is provenance, not a portable installation file. The primary numerical reproduction route uses saved predictions.

## Interpretation

Here a recording means a **file-level numerical unit**, not a verified independent acquisition session. Hashing excludes exactly identical numerical files; it cannot establish session independence. Five classes correspond to loaded states; the sixth is preset damage under no load, so damage and load are confounded. Three source rotational settings (159/318/540 RPM) correspond to 0.5/1.0/1.7 m/s. Hardware sampling frequency is unresolved in the source documentation; 400 Hz is only the released notebook's time-axis convention, and all experiment inputs are specified in samples.

All confidence intervals condition on the saved fitted models. Neither seeds nor windows are counted as additional independent file units. The split-gap interval uses the same class-stratified source-file multiplicities in both arms, retains all windows in each file, and preserves each arm's fold-mean scoring rule. Cohort intervals use matched target files and a Bonferroni adjustment for three speed comparisons. They do not include refitting, new-session or new-equipment variation. Most load–speed cells contain only two or three files, so speed-specific results are descriptive and exploratory.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for estimands, fixed seeds, output paths and limitations. Citation metadata are in [CITATION.cff](CITATION.cff).
