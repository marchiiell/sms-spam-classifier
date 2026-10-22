"""Helpers that write generated results into README.md.

The README contains HTML-comment markers such as ``<!-- RESULTS:START -->`` and
``<!-- RESULTS:END -->``. Everything between a pair of markers is replaced each
time the pipeline is run, so the numbers in the README always come from a real
run and are never typed in by hand.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def df_to_md(df: pd.DataFrame, floatfmt: str = ".3f", index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no extra deps)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]

    def fmt(v) -> str:
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return ""
        if isinstance(v, (float, np.floating)):
            return format(float(v), floatfmt)
        return str(v)

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def update_block(readme: Path, name: str, content: str) -> None:
    """Replace the text between ``<!-- NAME:START -->`` and ``<!-- NAME:END -->``."""
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(<!-- {re.escape(name)}:START -->)(.*?)(<!-- {re.escape(name)}:END -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(f"Markers for block '{name}' not found in {readme}")
    new_text = pattern.sub(
        lambda m: f"{m.group(1)}\n{content.strip()}\n{m.group(3)}", text, count=1
    )
    readme.write_text(new_text, encoding="utf-8")
