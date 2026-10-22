"""Classify messages with the saved pipeline.

Usage:
    python -m src.predict "Congratulations! You won a free prize, call now"
    python -m src.predict --file messages.txt          # one message per line
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import ROOT
from .models import get_scores


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("messages", nargs="*", help="message text(s) to classify")
    ap.add_argument("--file", help="text file with one message per line")
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"),
                    help="folder with spam_pipeline.joblib and metrics.json")
    args = ap.parse_args(argv)

    texts = list(args.messages)
    if args.file:
        texts += [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not texts:
        ap.error("provide at least one message or --file")

    folder = Path(args.reports_dir)
    model = joblib.load(folder / "spam_pipeline.joblib")
    threshold = json.loads((folder / "metrics.json").read_text())["threshold"]

    scores = get_scores(model, pd.Series(texts))
    for text, s in zip(texts, scores):
        label = "SPAM" if s >= threshold else "ham "
        print(f"{label}  score={s:+.3f}  {text[:90]}")


if __name__ == "__main__":
    main()
