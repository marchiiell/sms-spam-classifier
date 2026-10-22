"""Text normalisation and hand-crafted features."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_URL = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|co\.uk|info)\b\S*)", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"\b\d{5,}\b")          # long digit runs: phone numbers, short codes
_NUMBER = re.compile(r"\b\d+\b")
_CURRENCY = re.compile(r"[£$€]")

META_NAMES = [
    "log_chars", "log_words", "upper_ratio", "digit_ratio",
    "log_exclaim", "has_url", "has_currency", "has_long_number",
]


def normalize_text(s: str) -> str:
    """Lower-case and replace volatile strings with placeholder tokens.

    URLs, e-mails, phone/short-code numbers, other numbers and currency symbols
    are replaced by ``_url_``, ``_email_``, ``_phone_``, ``_num_`` and ``_cur_`` so
    the model learns *that* a message contains a number or link, not which one.
    """
    s = _EMAIL.sub(" _email_ ", s)   # before URLs: "me@x.org" also looks like a bare domain
    s = _URL.sub(" _url_ ", s)
    s = _CURRENCY.sub(" _cur_ ", s)
    s = _PHONE.sub(" _phone_ ", s)
    s = _NUMBER.sub(" _num_ ", s)
    return s.lower()


def meta_features(texts) -> np.ndarray:
    """Simple stylistic statistics of each message (n_samples x 8)."""
    rows = []
    for t in pd.Series(texts).astype(str):
        n = max(len(t), 1)
        letters = [c for c in t if c.isalpha()]
        rows.append([
            np.log1p(len(t)),
            np.log1p(len(t.split())),
            sum(c.isupper() for c in letters) / max(len(letters), 1),
            sum(c.isdigit() for c in t) / n,
            np.log1p(t.count("!")),
            float(bool(_URL.search(t))),
            float(bool(_CURRENCY.search(t))),
            float(bool(_PHONE.search(t))),
        ])
    return np.asarray(rows, dtype=float)
