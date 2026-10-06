"""Reconstruct file units, hashes, 400-sample windows and fixed fold allocation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {0: "0.5 kg load", 1: "1 kg load", 2: "2 kg load", 3: "3 kg load", 5: "5 kg load", 6: "damaged, unloaded"}


def allocate_folds(metadata, ids, n=3):
    """Original allocator: seeded stratum permutations, offsets carried across speeds."""
    metadata = metadata[metadata.recording_id.isin(ids)]
    result = [[] for _ in range(n)]
    rng = np.random.default_rng(20260908)
    offsets = {}
    for (condition, speed), group in metadata.groupby(["condition_encoded", "speed_rpm"]):
        names = rng.permutation(group.recording_id.to_numpy())
        offset = offsets.get(condition, 0)
        for j, name in enumerate(names):
            result[(j + offset) % n].append(str(name))
        offsets[condition] = (offset + len(names)) % n
    return result


def obtain_source(data_dir, manifest):
    data_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/ArmantasPik/Conveyor-belt-state-classification/zip/{manifest['commit']}"
    request = urllib.request.Request(url, headers={"User-Agent": "SIBIRCON-reproducibility"})
    with urllib.request.urlopen(request, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for key, expected in manifest["file_sha256"].items():
        name = key.split("/")[-1]
        matches = [entry for entry in archive.namelist() if entry.endswith("/RawData/" + name)]
        assert len(matches) == 1, name
        payload = archive.read(matches[0])
        assert hashlib.sha256(payload).hexdigest() == expected, name
        target = data_dir / name
        if target.exists():
            assert target.read_bytes() == payload, f"Refusing to overwrite a different file: {target}"
        else:
            target.write_bytes(payload)


def reconstruct(data_dir, output, download=False):
    manifest = json.loads((ROOT / "reference/source_manifest.json").read_text())
    if download:
        obtain_source(data_dir, manifest)
    files = sorted(data_dir.glob("*.csv"))
    assert len(files) == 79, f"Expected exactly 79 source CSV files, found {len(files)} in {data_dir}"
    rows = []
    arrays = {}
    for path in files:
        condition, speed, repetition = map(int, path.stem.split("_"))
        assert condition in LABELS and speed in [159, 318, 540] and repetition > 0
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest["file_sha256"]["RawData/" + path.name], path.name
        values = pd.read_csv(path).apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        assert values.ndim == 2 and values.shape[1] == 3 and np.isfinite(values).all()
        numeric_digest = hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()
        arrays[path.stem] = values
        rows.append(dict(recording_id=path.stem, condition_raw=condition, condition_encoded=sorted(LABELS).index(condition), condition_name=LABELS[condition], speed_rpm=speed, repetition=repetition, n_samples=len(values), file_sha256=digest, numeric_sha256=numeric_digest))
    full = pd.DataFrame(rows)
    metadata = full.drop_duplicates("numeric_sha256", keep="first").reset_index(drop=True)
    assert len(metadata) == 78
    reference = pd.read_csv(ROOT / "reference/metadata/recording_metadata.csv")
    assert full.recording_id.tolist() == reference.recording_id.tolist()
    assert full.numeric_sha256.tolist() == reference.numeric_sha256.tolist()
    windows, window_rows = [], []
    for record in metadata.itertuples():
        values = arrays[record.recording_id]
        for start in range(0, len(values) - 400 + 1, 400):
            windows.append(values[start:start + 400])
            window_rows.append(dict(window_id=f"{record.recording_id}_400_{start}", recording_id=record.recording_id, condition=record.condition_encoded, speed_rpm=record.speed_rpm, repetition=record.repetition, start_idx=start, end_idx=start + 400))
    wm = pd.DataFrame(window_rows)
    assert len(wm) == 509 and wm.recording_id.nunique() == 78
    pd.testing.assert_frame_equal(wm, pd.read_csv(ROOT / "reference/metadata/windows_400.csv"))
    splits = []
    ids = metadata.recording_id.tolist()
    for fold, test in enumerate(allocate_folds(metadata, ids)):
        train = sorted(set(ids) - set(test))
        splits.append(dict(name=f"mixed_{fold}", protocol="mixed_speed", held_out_speed=0, train=train, test=test, inner=allocate_folds(metadata, train)))
    assert splits == json.loads((ROOT / "reference/metadata/splits_mixed.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    full.to_csv(output / "source_file_metadata.csv", index=False)
    metadata.to_csv(output / "file_metadata.csv", index=False)
    wm.to_csv(output / "windows_400.csv", index=False)
    np.savez_compressed(output / "windows_400.npz", x=np.asarray(windows, dtype=np.float32))
    (output / "splits_mixed.json").write_text(json.dumps(splits, indent=2))
    duplicates = full[full.duplicated("numeric_sha256", keep=False)]
    duplicates.to_csv(output / "numeric_duplicates.csv", index=False)
    table = pd.crosstab(metadata.condition_encoded, metadata.speed_rpm).reindex(index=range(6), columns=[159, 318, 540], fill_value=0)
    table.index = list(LABELS.values())
    table.columns = ["0.5 m/s", "1.0 m/s", "1.7 m/s"]
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)
    table.to_csv(output / "table_I.csv")
    print(f"Verified 79 source files -> 78 numerical file units -> 509 windows; folds match release manifests. Output: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/RawData")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reconstruction")
    parser.add_argument("--download", action="store_true", help="Download only the 79 CSV files from the pinned upstream commit")
    args = parser.parse_args()
    reconstruct(args.data_dir, args.output, args.download)
