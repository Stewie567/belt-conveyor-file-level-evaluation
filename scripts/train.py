"""Retrain only the two FCN experiments reported in the conference paper."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.networks import Network
from src.evaluation.metrics import aggregate, metrics
from reconstruct import allocate_folds

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PC = [f"p{i}" for i in range(6)]


def fit(x, y, seed, epochs=60, validation=None):
    # Preserve the original FCN initialization, batch permutations and optimizer.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = Network("FCN", x.shape[2], 3).to(DEVICE)
    xt = torch.from_numpy(x.transpose(0, 2, 1).copy()).to(DEVICE)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    counts = np.bincount(y, minlength=6)
    assert (counts > 0).all()
    weights = torch.tensor(len(y) / (6 * counts), dtype=torch.float32, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    best, best_epoch, stale = -1., 1, 0
    history = []
    if validation is not None:
        vx, vy = validation
        vxt = torch.from_numpy(vx.transpose(0, 2, 1).copy()).to(DEVICE)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.
        for ix in torch.randperm(len(xt), device=DEVICE).split(64):
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(xt[ix], 0.)
            loss = criterion(logits, yt[ix])
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item() * len(ix)
        row = dict(epoch=epoch, train_loss=loss_sum / len(xt))
        if validation is not None:
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(v)[0] for v in vxt.split(64)]).argmax(1).cpu().numpy()
            score = f1_score(vy, pred, labels=range(6), average="macro", zero_division=0)
            row["validation_macro_f1"] = float(score)
            if score > best + 1e-10:
                best, best_epoch, stale = float(score), epoch, 0
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= 12:
            break
    return model, best_epoch if validation is not None else epochs, history


def raw_scaler(data_dir, ids):
    values = np.concatenate([pd.read_csv(data_dir / f"{name}.csv").to_numpy(dtype=np.float64) for name in sorted(set(ids))])
    return values.mean(0), np.maximum(values.std(0), 1e-8)


def standardized(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def select_budget(x, windows, metadata, train_ids, data_dir, seed):
    chosen, evidence = [], []
    for fold, valid_ids in enumerate(allocate_folds(metadata, train_ids)):
        inner_ids = sorted(set(train_ids) - set(valid_ids))
        assert not set(inner_ids) & set(valid_ids)
        train = windows.recording_id.isin(inner_ids).to_numpy()
        valid = windows.recording_id.isin(valid_ids).to_numpy()
        mean, std = raw_scaler(data_dir, inner_ids)
        _, epoch, history = fit(standardized(x[train], mean, std), windows.loc[train, "condition"].to_numpy(), seed, validation=(standardized(x[valid], mean, std), windows.loc[valid, "condition"].to_numpy()))
        chosen.append(epoch)
        evidence.append(dict(fold=fold, best_epoch=epoch, train_files=inner_ids, validation_files=valid_ids, scaler_mean=mean.tolist(), scaler_std=std.tolist(), history=history))
    return max(1, int(np.median(chosen))), evidence


def predict(model, x):
    model.eval()
    result = []
    with torch.no_grad():
        for values in np.array_split(x, max(1, int(np.ceil(len(x) / 64)))):
            logits, _, _ = model(torch.from_numpy(values.transpose(0, 2, 1).copy()).to(DEVICE))
            result.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(result)


def save_run(folder, x, windows, train, test, seed, epochs, mean, std, config, selection):
    assert not set(windows.loc[train, "window_id"]) & set(windows.loc[test, "window_id"])
    config = dict(config, seed=seed, epochs=epochs, scaler_mean=mean.tolist(), scaler_std=std.tolist(), train_windows=windows.loc[train, "window_id"].tolist(), test_windows=windows.loc[test, "window_id"].tolist())
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (folder / "complete.json").exists():
        marker = json.loads((folder / "complete.json").read_text())
        assert marker["configuration_sha256"] == signature, "Use a fresh output directory after changing settings."
        for filename, digest in marker["files"].items():
            assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest, filename
        return pd.read_csv(folder / "window_predictions.csv")
    folder.mkdir(parents=True, exist_ok=True)
    model, _, history = fit(standardized(x[train], mean, std), windows.loc[train, "condition"].to_numpy(), seed, epochs=epochs)
    probabilities = predict(model, standardized(x[test], mean, std))
    rows = windows.loc[test].copy()
    for column, values in zip(PC, probabilities.T):
        rows[column] = values
    files = aggregate(rows, probabilities)
    rows.to_csv(folder / "window_predictions.csv", index=False)
    files.to_csv(folder / "recording_predictions.csv", index=False)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    (folder / "inner_selection.json").write_text(json.dumps(selection, indent=2))
    (folder / "metrics.json").write_text(json.dumps(dict(window=metrics(rows.condition, probabilities), recording=metrics(files.condition, files[PC].to_numpy())), indent=2))
    pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
    torch.save(model.state_dict(), folder / "model.pt")
    digest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.iterdir() if path.is_file() and path.name != "complete.json"}
    (folder / "complete.json").write_text(json.dumps(dict(configuration_sha256=signature, files=digest), indent=2))
    print("Finished", folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["split", "cohort", "all", "smoke"], default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/RawData")
    parser.add_argument("--reconstruction", type=Path, default=ROOT / "outputs/reconstruction")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/training")
    args = parser.parse_args()
    if args.task == "smoke":
        x = np.random.default_rng(0).normal(size=(12, 400, 3)).astype(np.float32)
        y = np.repeat(np.arange(6), 2)
        model, epoch, history = fit(x, y, 0, epochs=1)
        probabilities = predict(model, x)
        assert probabilities.shape == (12, 6) and np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(1), 1., atol=1e-6)
        print("FCN training and prediction smoke test passed on", DEVICE)
        return
    x = np.load(args.reconstruction / "windows_400.npz")["x"]
    windows = pd.read_csv(args.reconstruction / "windows_400.csv")
    metadata = pd.read_csv(args.reconstruction / "file_metadata.csv")
    splits = json.loads((args.reconstruction / "splits_mixed.json").read_text())
    assert len(x) == len(windows) == 509 and windows.recording_id.nunique() == 78
    if args.task in ["split", "all"]:
        window_folds = list(StratifiedKFold(3, shuffle=True, random_state=20260908).split(x, windows.condition))
        for seed in range(3):
            for fold, split in enumerate(splits):
                for arm in ["window", "recording"]:
                    if arm == "window":
                        train_ix, test_ix = window_folds[fold]
                        train = np.zeros(len(x), dtype=bool); train[train_ix] = True
                        test = np.zeros(len(x), dtype=bool); test[test_ix] = True
                    else:
                        train = windows.recording_id.isin(split["train"]).to_numpy()
                        test = windows.recording_id.isin(split["test"]).to_numpy()
                    mean = x[train].mean((0, 1), dtype=np.float64)
                    std = np.maximum(x[train].std((0, 1), dtype=np.float64), 1e-8)
                    save_run(args.output / f"MR-01/{arm}/f{fold}_s{seed}", x, windows, train, test, seed, 60, mean, std, dict(task="MR-01", arm=arm, fold=fold, normalization="training windows only"), [])
    if args.task in ["cohort", "all"]:
        for seed in range(3):
            for fold, split in enumerate(splits):
                seen_folder = args.output / f"seen_models/f{fold}_s{seed}"
                seen_train = windows.recording_id.isin(split["train"]).to_numpy()
                outer_test = windows.recording_id.isin(split["test"]).to_numpy()
                mean, std = raw_scaler(args.data_dir, split["train"])
                if (seen_folder / "config.json").exists() and (seen_folder / "complete.json").exists():
                    config = json.loads((seen_folder / "config.json").read_text())
                    epochs = config["epochs"]
                    selection = json.loads((seen_folder / "inner_selection.json").read_text())
                else:
                    epochs, selection = select_budget(x, windows, metadata, split["train"], args.data_dir, seed)
                seen_rows = save_run(seen_folder, x, windows, seen_train, outer_test, seed, epochs, mean, std, dict(task="seen model", fold=fold, normalization="permitted raw files including tails"), selection)
                for speed in [159, 318, 540]:
                    seen = seen_rows[seen_rows.speed_rpm == speed].copy()
                    folder = args.output / f"MR-02/{speed}/seen_f{fold}_s{seed}"
                    folder.mkdir(parents=True, exist_ok=True)
                    seen.to_csv(folder / "window_predictions.csv", index=False)
                    aggregate(seen, seen[PC].to_numpy()).to_csv(folder / "recording_predictions.csv", index=False)
                    ids = metadata.loc[metadata.recording_id.isin(split["train"]) & (metadata.speed_rpm != speed), "recording_id"].tolist()
                    assert not set(ids) & set(split["test"])
                    train = windows.recording_id.isin(ids).to_numpy()
                    test = outer_test & (windows.speed_rpm.to_numpy() == speed)
                    unseen_folder = args.output / f"MR-02/{speed}/unseen_f{fold}_s{seed}"
                    if (unseen_folder / "config.json").exists() and (unseen_folder / "complete.json").exists():
                        saved = json.loads((unseen_folder / "config.json").read_text())
                        epochs_unseen = saved["epochs"]
                        selection_unseen = json.loads((unseen_folder / "inner_selection.json").read_text())
                    else:
                        epochs_unseen, selection_unseen = select_budget(x, windows, metadata, ids, args.data_dir, seed)
                    unseen_mean, unseen_std = raw_scaler(args.data_dir, ids)
                    save_run(unseen_folder, x, windows, train, test, seed, epochs_unseen, unseen_mean, unseen_std, dict(task="MR-02", arm="unseen", speed_rpm=speed, fold=fold, normalization="non-target raw files including tails"), selection_unseen)


if __name__ == "__main__":
    main()
