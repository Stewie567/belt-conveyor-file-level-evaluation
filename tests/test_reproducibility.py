"""Tests for the evaluation unit and estimands, not arbitrary output snapshots."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reproduce
import reconstruct


class EvaluationTests(unittest.TestCase):
    def test_evidence_integrity(self):
        self.assertGreaterEqual(reproduce.verify_evidence(), 500)

    def test_zero_support_classes_remain_in_macro_average(self):
        cm = np.zeros((6, 6)); cm[0, 0] = 10
        self.assertAlmostEqual(float(reproduce.macro_f1(cm)), 1 / 6)

    def test_file_contributions_keep_all_windows(self):
        rows = pd.DataFrame({"recording_id": ["a", "a", "b"], "condition": [0, 0, 1]})
        for c in range(6):
            rows[f"p{c}"] = [float(c == 0), float(c == 1), float(c == 1)]
        contributions = reproduce.file_contributions(rows, ["a", "b"])
        self.assertEqual(contributions[0].sum(), 2)
        self.assertEqual(contributions[1].sum(), 1)
        self.assertTrue(np.array_equal(contributions.sum(0), reproduce.confusion(rows)))

    def test_class_stratified_resampling_preserves_class_totals(self):
        labels = np.repeat(np.arange(6), [2, 3, 3, 3, 3, 11])
        weights = reproduce.resample_counts(np.random.default_rng(0), labels, 100)
        for c in range(6):
            self.assertTrue(np.all(weights[:, labels == c].sum(1) == (labels == c).sum()))

    def test_fold_allocator_has_no_file_overlap(self):
        m = pd.read_csv(ROOT / "reference/metadata/recording_metadata.csv").drop_duplicates("numeric_sha256")
        folds = reconstruct.allocate_folds(m, m.recording_id.tolist())
        self.assertEqual(sum(map(len, folds)), 78)
        for i in range(3):
            for j in range(i):
                self.assertFalse(set(folds[i]) & set(folds[j]))
            self.assertEqual(set(m[m.recording_id.isin(folds[i])].condition_encoded), set(range(6)))

    def test_fold_mean_differs_from_pooled_metric(self):
        a = np.eye(6) * 2
        b = np.zeros((6, 6)); b[:, 0] = 100
        mean = (reproduce.macro_f1(a) + reproduce.macro_f1(b)) / 2
        pooled = reproduce.macro_f1(a + b)
        self.assertGreater(abs(mean - pooled), .1)


if __name__ == "__main__":
    unittest.main()
