"""Recompute paper tables and conditional bootstrap intervals from saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reference/predictions"
PC = [f"p{i}" for i in range(6)]
SPEEDS = {159: 0.5, 318: 1.0, 540: 1.7}


def macro_f1(cm):
    denominator = cm.sum(axis=-1) + cm.sum(axis=-2)
    return np.divide(2 * np.diagonal(cm, axis1=-2, axis2=-1), denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0).mean(axis=-1)


def confusion(rows):
    true = rows.condition.to_numpy(dtype=int)
    pred = rows[PC].to_numpy().argmax(axis=1)
    return np.bincount(true * 6 + pred, minlength=36).reshape(6, 6)


def file_contributions(rows, ids):
    """One confusion contribution per source file; all its windows stay together."""
    result = np.zeros((len(ids), 6, 6), dtype=float)
    index = {name: i for i, name in enumerate(ids)}
    for name, group in rows.groupby("recording_id"):
        result[index[name]] = confusion(group)
    return result


def resample_counts(rng, labels, replicates):
    weights = np.zeros((replicates, len(labels)), dtype=np.int16)
    for condition in range(6):
        ix = np.flatnonzero(labels == condition)
        assert len(ix) > 0
        weights[:, ix] = rng.multinomial(len(ix), np.ones(len(ix)) / len(ix), size=replicates)
    return weights


def verify_evidence():
    evidence = json.loads((ROOT / "reference/SHA256.json").read_text())
    for relative, expected in evidence.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    return len(evidence)


def split_comparison(output, replicates):
    metadata = pd.read_csv(ROOT / "reference/metadata/recording_metadata.csv").drop_duplicates("numeric_sha256").reset_index(drop=True)
    ids = metadata.recording_id.tolist()
    labels = metadata.condition_encoded.to_numpy()
    # axes: file, arm, training seed, outer fold, true class, predicted class
    contributions = np.zeros((78, 2, 3, 3, 6, 6), dtype=float)
    rows_out, count_rows = [], []
    manifest = json.loads((ROOT / "reference/metadata/splits_mixed.json").read_text())
    for arm_index, arm in enumerate(["window", "recording"]):
        for seed in range(3):
            scores = []
            seen_windows = []
            for fold in range(3):
                folder = PREDICTIONS / f"MR-01/{arm}/f{fold}_s{seed}"
                rows = pd.read_csv(folder / "window_predictions.csv")
                config = json.loads((folder / "config.json").read_text())
                assert config["epochs"] == 60 and config["normalization"] in ["windows", "training windows only"]
                assert set(rows.window_id) == set(config["test_windows"])
                assert not set(config["train_windows"]) & set(config["test_windows"])
                if arm == "recording":
                    assert set(rows.recording_id) == set(manifest[fold]["test"])
                seen_windows.extend(rows.window_id)
                contributions[:, arm_index, seed, fold] = file_contributions(rows, ids)
                cm = confusion(rows)
                score = dict(macro_f1=float(macro_f1(cm)), accuracy=float(np.trace(cm) / cm.sum()), balanced_accuracy=float(np.divide(np.diag(cm), cm.sum(1), out=np.zeros(6), where=cm.sum(1) != 0).mean()))
                scores.append(score)
                if seed == 0:
                    unique = rows.drop_duplicates("recording_id")
                    for c in range(6):
                        count_rows.append(dict(split=arm, fold=fold + 1, condition=c, state=list(metadata.condition_name.drop_duplicates())[c], test_files=int((unique.condition == c).sum()), test_windows=int((rows.condition == c).sum())))
            assert len(seen_windows) == 509 and len(set(seen_windows)) == 509
            rows_out.append(dict(split=arm, seed=seed, **{name: np.mean([score[name] for score in scores]) for name in scores[0]}))
    by_seed = pd.DataFrame(rows_out)
    by_seed.to_csv(output / "split_by_seed.csv", index=False)
    by_seed.groupby("split")[["macro_f1", "accuracy", "balanced_accuracy"]].agg(["mean", "std"]).to_csv(output / "table_II.csv")
    pd.DataFrame(count_rows).to_csv(output / "test_counts_by_fold.csv", index=False)
    point = macro_f1(contributions.sum(axis=0)).mean(axis=-1).mean(axis=-1)
    if PREDICTIONS == ROOT / "reference/predictions":
        assert np.allclose(point, [0.5584, 0.2950], atol=0.00005), point
    differences = []
    rng = np.random.default_rng(20261007)
    flattened = contributions.reshape(78, -1)
    for start in range(0, replicates, 1000):
        weights = resample_counts(rng, labels, min(1000, replicates - start))
        cm = (weights @ flattened).reshape(-1, 2, 3, 3, 6, 6)
        # Keep the original estimand: average three folds within each seed,
        # then average three training seeds. Do not pool folds or count seeds as files.
        scores = macro_f1(cm).mean(axis=-1).mean(axis=-1)
        differences.extend(scores[:, 0] - scores[:, 1])
    low, high = np.quantile(differences, [0.025, 0.975])
    result = dict(window_macro_f1=float(point[0]), file_disjoint_macro_f1=float(point[1]), gap=float(point[0] - point[1]), ci_low=float(low), ci_high=float(high), replicates=replicates, random_seed=20261007, unit="paired source-file clusters, stratified by class; shared multiplicities across arms and training seeds", estimand="mean of three outer-fold Macro-F1 values per seed, averaged across three seeds", limitations="conditional on fitted models and observed fold assignments; no model refits or session resampling; not a pure leakage effect")
    (output / "split_gap_interval.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame({"gap": differences}).to_csv(output / "split_gap_bootstrap.csv", index=False)
    return result


def cohort_comparison(output, replicates):
    result, bootstrap_summary = [], []
    for speed, linear in SPEEDS.items():
        contributions = None
        cohort_ids = None
        labels = None
        for seed in range(3):
            arm_rows = {}
            for arm_index, arm in enumerate(["seen", "unseen"]):
                for level in ["window", "recording"]:
                    tables = [pd.read_csv(PREDICTIONS / f"MR-02/{speed}/{arm}_f{fold}_s{seed}/{level}_predictions.csv") for fold in range(3)]
                    rows = pd.concat(tables, ignore_index=True).sort_values("window_id" if level == "window" else "recording_id")
                    key = "window_id" if level == "window" else "recording_id"
                    assert rows[key].is_unique
                    arm_rows[(arm, level)] = rows
                    result.append(dict(speed_rpm=speed, speed_ms=linear, level=level, seed=seed, arm=arm, macro_f1=float(macro_f1(confusion(rows))), n=len(rows)))
                    if level == "recording":
                        ids = rows.recording_id.tolist()
                        y = rows.condition.to_numpy()
                        if contributions is None:
                            cohort_ids, labels = ids, y
                            contributions = np.zeros((len(ids), 2, 3, 6, 6))
                        assert ids == cohort_ids and np.array_equal(y, labels)
                        contributions[:, arm_index, seed] = file_contributions(rows, ids)
            for level in ["window", "recording"]:
                key = "window_id" if level == "window" else "recording_id"
                assert arm_rows[("seen", level)][key].tolist() == arm_rows[("unseen", level)][key].tolist()
        differences = []
        rng = np.random.default_rng(20261007 + speed)
        flattened = contributions.reshape(len(labels), -1)
        for start in range(0, replicates, 1000):
            weights = resample_counts(rng, labels, min(1000, replicates - start))
            cm = (weights @ flattened).reshape(-1, 2, 3, 6, 6)
            scores = macro_f1(cm).mean(axis=-1)
            differences.extend(scores[:, 0] - scores[:, 1])
        low, high = np.quantile(differences, [.025, .975])
        adjusted_low, adjusted_high = np.quantile(differences, [.05 / 6, 1 - .05 / 6])
        point = macro_f1(contributions.sum(0)).mean(axis=-1)
        bootstrap_summary.append(dict(speed_rpm=speed, speed_ms=linear, seen=point[0], unseen=point[1], delta=point[0] - point[1], unadjusted_low=low, unadjusted_high=high, adjusted_low=adjusted_low, adjusted_high=adjusted_high, target_files=len(labels)))
        pd.DataFrame({"delta": differences}).to_csv(output / f"cohort_bootstrap_{speed}.csv", index=False)
    by_seed = pd.DataFrame(result)
    by_seed.to_csv(output / "cohort_by_seed.csv", index=False)
    window = by_seed[by_seed.level == "window"].groupby(["speed_rpm", "arm"]).macro_f1.mean().unstack()
    final = pd.DataFrame(bootstrap_summary)
    final["window_delta"] = final.speed_rpm.map(window.seen - window.unseen)
    final.to_csv(output / "table_III.csv", index=False)
    return final.to_dict("records")


def main():
    global PREDICTIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis")
    parser.add_argument("--replicates", type=int, default=50000)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS, help="Directory containing MR-01 and MR-02; defaults to published reference evidence")
    args = parser.parse_args()
    PREDICTIONS = args.predictions.resolve()
    assert args.replicates >= 1000
    args.output.mkdir(parents=True, exist_ok=True)
    files = verify_evidence()
    gap = split_comparison(args.output, args.replicates)
    cohorts = cohort_comparison(args.output, args.replicates)
    summary = dict(evidence_files_verified=files, bootstrap_replicates=args.replicates, split_comparison=gap, cohort_comparison=cohorts, multiplicity="Bonferroni: three equal-tailed 98.333% marginal percentile intervals, nominal family coverage 95%", caveat="All intervals are approximate, conditional on saved fitted models; file identity does not imply independent acquisition sessions.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
