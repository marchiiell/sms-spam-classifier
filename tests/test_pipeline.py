"""Unit tests. Run with:  python -m unittest   (pytest also works)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import _read_file, clean, make_demo_data
from src.models import get_models, get_scores
from src.text import META_NAMES, meta_features, normalize_text
from src.train import threshold_for_precision


class TextTests(unittest.TestCase):
    def test_normalize_replaces_volatile_tokens(self):
        out = normalize_text("Call 08001234567 now! Win £500 at www.prize.com or mail me@x.org, room 12")
        for token in ("_phone_", "_cur_", "_url_", "_email_", "_num_"):
            self.assertIn(token, out)
        self.assertNotIn("08001234567", out)
        self.assertEqual(out, out.lower())

    def test_meta_features_shape_and_flags(self):
        m = meta_features(["FREE!!! Visit www.win.com for £100 now 8001234", "ok see you"])
        self.assertEqual(m.shape, (2, len(META_NAMES)))
        idx = {n: i for i, n in enumerate(META_NAMES)}
        self.assertEqual(m[0, idx["has_url"]], 1.0)
        self.assertEqual(m[0, idx["has_currency"]], 1.0)
        self.assertEqual(m[0, idx["has_long_number"]], 1.0)
        self.assertEqual(m[1, idx["has_url"]], 0.0)
        self.assertGreater(m[0, idx["upper_ratio"]], m[1, idx["upper_ratio"]])

    def test_meta_features_handles_empty_text(self):
        self.assertTrue(np.isfinite(meta_features([""])).all())


class DataTests(unittest.TestCase):
    def test_clean_encodes_labels_and_drops_duplicates(self):
        raw = pd.DataFrame({"label": ["ham", "spam", "ham", "HAM "],
                            "text": ["hello  there", "win now", "hello there", "other"]})
        df = clean(raw)
        self.assertEqual(len(df), 3)  # "hello  there" == "hello there" after whitespace fix
        self.assertEqual(set(df["label"]), {0, 1})

    def test_clean_rejects_unknown_labels(self):
        with self.assertRaises(ValueError):
            clean(pd.DataFrame({"label": ["maybe"], "text": ["x"]}))

    def test_read_tab_separated_file_with_quotes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sms.tsv"
            p.write_text('ham\tHe said "hi" to me\nspam\tWIN a prize\n', encoding="utf-8")
            df = _read_file(p)
        self.assertEqual(list(df.columns), ["label", "text"])
        self.assertEqual(df.loc[0, "text"], 'He said "hi" to me')


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = clean(make_demo_data(n=600, seed=1))
        cls.X, cls.y = df["text"], df["label"]

    def test_every_model_fits_and_scores(self):
        for name, model in get_models(seed=0).items():
            with self.subTest(model=name):
                model.fit(self.X, self.y)
                scores = get_scores(model, self.X)
                self.assertEqual(scores.shape, (len(self.X),))
                self.assertTrue(np.isfinite(scores).all())

    def test_text_model_separates_demo_classes(self):
        model = get_models(seed=0)["TF-IDF + Logistic regression"].fit(self.X, self.y)
        spam = model.predict(pd.Series(["WINNER!! claim your free prize now, call 81234"]))[0]
        ham = model.predict(pd.Series(["can you call me when you get home"]))[0]
        self.assertEqual((spam, ham), (1, 0))


class ThresholdTests(unittest.TestCase):
    def test_precision_target_reached_with_max_recall(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.6, 0.5, 0.7, 0.8, 0.9])  # one ham scores 0.6
        thr, reached = threshold_for_precision(y, s, target=1.0)
        self.assertTrue(reached)
        self.assertEqual(thr, 0.7)           # lowest cut-off with zero false positives
        thr90, _ = threshold_for_precision(y, s, target=0.75)
        self.assertEqual(thr90, 0.5)         # allows the ham at 0.6 -> higher recall

    def test_unreachable_target_falls_back(self):
        y = np.array([1, 0, 1, 0])
        s = np.array([0.2, 0.9, 0.3, 0.8])    # scores are anti-correlated with labels
        _, reached = threshold_for_precision(y, s, target=0.99)
        self.assertFalse(reached)


if __name__ == "__main__":
    unittest.main()
