"""Loading and cleaning the SMS Spam Collection.

Real data
---------
`SMS Spam Collection <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>`_
(UCI): 5,574 English SMS messages labelled ``ham`` or ``spam``.

Either place a copy in ``data/raw/`` (the original tab-separated
``SMSSpamCollection``, a ``sms.tsv`` or the Kaggle ``spam.csv`` all work) or just
run the pipeline - ``load_raw`` downloads it on the first run.

Demo data
---------
``make_demo_data`` builds *synthetic* template-based messages so that the code
and tests run offline. Scores on it are meaningless.
"""
from __future__ import annotations

import csv
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TSV_URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
CANDIDATE_NAMES = ["SMSSpamCollection", "SMSSpamCollection.tsv", "sms.tsv", "spam.csv"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":  # Kaggle version: columns v1 (label), v2 (text)
        df = pd.read_csv(path, encoding="latin-1")
        df = df.iloc[:, :2]
    else:  # original: "label<TAB>text", no header, quotes are part of the text
        df = pd.read_csv(path, sep="\t", header=None, quoting=csv.QUOTE_NONE,
                         encoding="utf-8", encoding_errors="replace")
    df.columns = ["label", "text"]
    return df


def _download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "sms.tsv"
    try:
        urllib.request.urlretrieve(TSV_URL, target)
        return target
    except Exception:  # noqa: BLE001 - fall back to the official UCI archive
        pass
    with urllib.request.urlopen(ZIP_URL, timeout=60) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        target = RAW_DIR / "SMSSpamCollection"
        target.write_bytes(zf.read("SMSSpamCollection"))
    return target


def load_raw(demo: bool = False) -> pd.DataFrame:
    """Return the raw table with columns ``label`` and ``text``."""
    if demo:
        return make_demo_data()
    for name in CANDIDATE_NAMES:
        if (RAW_DIR / name).exists():
            return _read_file(RAW_DIR / name)
    print("Downloading the SMS Spam Collection ...")
    try:
        return _read_file(_download())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not download the dataset. Download it manually (see README) "
            f"and place it in {RAW_DIR}. Original error: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise labels/whitespace and drop duplicate messages.

    The raw collection contains hundreds of exact duplicates. Left in, copies of
    the same message land in both train and test folds and inflate every score,
    so they are removed *before* any split.
    """
    out = df[["label", "text"]].dropna().copy()
    out["label"] = out["label"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1})
    if out["label"].isna().any():
        raise ValueError("Unexpected label values (expected 'ham' / 'spam').")
    out["label"] = out["label"].astype(int)
    out["text"] = out["text"].astype(str).map(lambda s: re.sub(r"\s+", " ", s).strip())
    out = out[out["text"] != ""]
    return out.drop_duplicates(subset="text").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Synthetic stand-in (offline demo / tests only)
# --------------------------------------------------------------------------- #
_HAM = [
    "ok see you at {t}", "can you call me when you get home", "lol that was so funny",
    "i will be late, the bus is stuck", "did you finish the homework for {d}",
    "happy birthday! hope you have a great day", "what time is the meeting tomorrow",
    "thanks for dinner, it was lovely", "are we still on for lunch on {d}",
    "just woke up, give me ten minutes", "send me the notes from class please",
    "i am at the library, come over if you want", "good night, talk tomorrow",
    "mum says dinner is at {t}", "can you pick up some milk on your way back",
]
_SPAM = [
    "WINNER!! You have won a {n} prize. Call {p} now to claim your reward",
    "Free entry in our weekly comp. Text WIN to {p} to enter. T&Cs apply",
    "URGENT! Your mobile number has been selected for a {n} cash award. Call {p}",
    "Congratulations! Claim your free ringtone now, reply YES to {p}",
    "You have been chosen for a free holiday. Visit www.prize-{n}.com to claim",
    "Final notice: your account needs verification, click www.verify-{n}.net today",
    "Get cheap loans fast! Text LOAN to {p}. 18+ only. Free to reply",
    "Hot singles in your area. Txt CHAT to {p} now, {n} free minutes",
]
_DAYS = ["monday", "tuesday", "friday", "the weekend", "sunday"]
_FILLER = [
    "ok", "thanks", "pls", "now", "today", "soon", "really", "maybe", "again", "later",
    "yes", "no", "sure", "cool", "fine", "great", "sorry", "wait", "quick", "tonight",
    "home", "work", "town", "phone", "car", "bus", "train", "rain", "sun", "tea",
    "coffee", "lunch", "movie", "game", "music", "party", "friend", "sister", "brother", "dog",
]


def make_demo_data(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        is_spam = rng.random() < 0.13
        tpl = rng.choice(_SPAM if is_spam else _HAM)
        text = tpl.format(
            t=f"{rng.integers(1, 12)}:{rng.choice(['00', '15', '30', '45'])}",
            d=rng.choice(_DAYS), n=int(rng.integers(100, 5000)),
            p=str(int(rng.integers(80000, 89999))),
        )
        # noise: a few random filler words so messages are not all identical
        text += " " + " ".join(rng.choice(_FILLER, size=int(rng.integers(1, 4))))
        if not is_spam and rng.random() < 0.1:
            text = text.upper()
        rows.append(("spam" if is_spam else "ham", text))
    return pd.DataFrame(rows, columns=["label", "text"])
