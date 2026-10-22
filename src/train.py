"""Train, compare and evaluate SMS spam classifiers.

Usage:
    python -m src.train                  # real data; writes results into README.md
    python -m src.train --demo           # synthetic data (offline smoke test)
    python -m src.train --target-precision 0.995

Workflow
--------
1. Remove duplicate messages, then a stratified train/test split.
2. 5-fold stratified CV on the training set to compare models.
3. Business requirement: *a legitimate message must almost never be blocked.*
   For the best model, the decision threshold is chosen on out-of-fold training
   scores as the one with the highest recall subject to precision >= target.
4. One final evaluation on the test set, error analysis, interpretability.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    make_scorer, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from .data import ROOT, clean, load_raw
from .models import get_models, get_scores
from .plotting import (
    plot_confusion, plot_length_distribution, plot_pr_curves, plot_top_terms,
)
from .report import df_to_md, update_block
from .text import normalize_text


def threshold_for_precision(y_true, scores, target: float) -> tuple[float, bool]:
    """Lowest threshold (= highest recall) whose precision is >= ``target``.

    Returns ``(threshold, reached)``; if the target is unreachable the threshold
    with the highest precision is returned and ``reached`` is False.
    """
    prec, rec, thr = precision_recall_curve(y_true, scores)
    prec, rec = prec[:-1], rec[:-1]            # align with thresholds
    ok = np.where(prec >= target)[0]
    if len(ok) == 0:
        return float(thr[int(np.argmax(prec))]), False
    return float(thr[ok[int(np.argmax(rec[ok]))]]), True


def oof_scores(model, X, y, cv) -> np.ndarray:
    method = "predict_proba" if hasattr(model, "predict_proba") else "decision_function"
    out = cross_val_predict(model, X, y, cv=cv, method=method)
    return out[:, 1] if out.ndim == 2 else out


def top_terms(X_train, y_train, k: int = 15):
    """Interpretable word-level LR just for the 'top terms' plot."""
    vec = TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 1), min_df=3, sublinear_tf=True)
    lr = LogisticRegression(C=10, max_iter=2000).fit(vec.fit_transform(X_train), y_train)
    names, coef = np.array(vec.get_feature_names_out()), lr.coef_[0]
    order = np.argsort(coef)
    spam = [(names[i], coef[i]) for i in order[::-1][:k]]
    ham = [(names[i], coef[i]) for i in order[:k]]
    return spam, ham


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--target-precision", type=float, default=0.99,
                    help="minimum share of flagged messages that really are spam")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    out = ROOT / "reports" / ("demo" if args.demo else "")
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(demo=args.demo)
    df = clean(raw)
    removed = len(raw) - len(df)
    X, y = df["text"], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    print(f"Messages: {len(raw):,} raw -> {len(df):,} after removing {removed:,} duplicates/empty | "
          f"spam share {y.mean():.1%} | train {len(X_tr):,} / test {len(X_te):,}")

    cv = StratifiedKFold(args.cv, shuffle=True, random_state=args.seed)
    models = get_models(args.seed)

    # ---- 1. cross-validated model comparison ---- #
    rows = []
    scoring = {  # zero_division=0 keeps the majority-class baseline from spamming warnings
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "pr_auc": "average_precision", "roc_auc": "roc_auc"}
    for name, model in models.items():
        res = cross_validate(clone(model), X_tr, y_tr, cv=cv, scoring=scoring)
        rows.append({"model": name, **{k: res[f"test_{k}"].mean() for k in scoring},
                     "pr_auc_std": res["test_pr_auc"].std()})
        print(f"  {name:34s} F1 {rows[-1]['f1']:.3f}  PR-AUC {rows[-1]['pr_auc']:.3f}")
    cv_table = pd.DataFrame(rows)
    best_name = (cv_table[~cv_table["model"].str.startswith("Baseline")]
                 .sort_values("pr_auc", ascending=False).iloc[0]["model"])
    print(f"Best model by CV PR-AUC: {best_name}")

    # ---- 2. operating point from out-of-fold scores ---- #
    thr, reached = threshold_for_precision(
        y_tr, oof_scores(clone(models[best_name]), X_tr, y_tr, cv), args.target_precision)
    print(f"Threshold for precision >= {args.target_precision:.1%}: {thr:.3f} "
          f"({'reached' if reached else 'NOT reached - using max-precision point'})")

    # ---- 3. final fit, single test evaluation ---- #
    fitted = {n: clone(m).fit(X_tr, y_tr) for n, m in models.items()}
    test_scores = {n: get_scores(m, X_te) for n, m in fitted.items()}
    test_rows = []
    for n, m in fitted.items():
        pred = m.predict(X_te)
        test_rows.append({
            "model": n, "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
            "pr_auc": average_precision_score(y_te, test_scores[n]),
            "roc_auc": roc_auc_score(y_te, test_scores[n]),
        })
    test_table = pd.DataFrame(test_rows)

    s_best = test_scores[best_name]
    pred_default = fitted[best_name].predict(X_te)
    pred_tuned = (s_best >= thr).astype(int)
    ops = pd.DataFrame([
        {"operating point": "Default predict()", "precision": precision_score(y_te, pred_default, zero_division=0),
         "recall": recall_score(y_te, pred_default),
         "ham blocked (FP)": int(((pred_default == 1) & (y_te == 0)).sum()),
         "spam missed (FN)": int(((pred_default == 0) & (y_te == 1)).sum())},
        {"operating point": f"Precision >= {args.target_precision:.0%} threshold",
         "precision": precision_score(y_te, pred_tuned, zero_division=0),
         "recall": recall_score(y_te, pred_tuned),
         "ham blocked (FP)": int(((pred_tuned == 1) & (y_te == 0)).sum()),
         "spam missed (FN)": int(((pred_tuned == 0) & (y_te == 1)).sum())},
    ])

    # ---- 4. error analysis & interpretability ---- #
    errors = pd.DataFrame({"text": X_te.values, "label": y_te.values, "score": s_best,
                           "predicted": pred_tuned})
    errors = errors[errors["label"] != errors["predicted"]].copy()
    errors["error_type"] = np.where(errors["label"] == 0, "false positive (ham blocked)",
                                    "false negative (spam missed)")
    errors.sort_values("score").to_csv(out / "errors.csv", index=False)
    spam_terms, ham_terms = top_terms(X_tr, y_tr)

    # ---- 5. figures and artefacts ---- #
    plot_length_distribution(X, y, fig_dir / "length_distribution.png")
    plot_pr_curves(y_te, {n: s for n, s in test_scores.items() if not n.startswith("Baseline")},
                   fig_dir / "pr_curves.png")
    plot_confusion(confusion_matrix(y_te, pred_tuned),
                   f"{best_name}\nprecision >= {args.target_precision:.0%}", fig_dir / "confusion_matrix.png")
    plot_top_terms(spam_terms, ham_terms, fig_dir / "top_terms.png")

    joblib.dump(fitted[best_name], out / "spam_pipeline.joblib")
    cv_table.to_csv(out / "cv_model_comparison.csv", index=False)
    test_table.to_csv(out / "test_metrics.csv", index=False)
    summary = {"best_model": best_name, "threshold": thr, "precision_target": args.target_precision,
               "precision_target_reached_in_cv": reached, "messages": len(df),
               "duplicates_removed": removed, "spam_share": float(y.mean()),
               "seed": args.seed, "demo_data": args.demo}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- 6. README ---- #
    if not args.demo:
        n_fp = int(ops.iloc[1]["ham blocked (FP)"])
        n_fn = int(ops.iloc[1]["spam missed (FN)"])
        md = [
            f"_Generated by `python -m src.train` on {date.today():%Y-%m-%d} (seed {args.seed}): "
            f"{len(raw):,} raw messages, {removed:,} duplicates removed -> {len(df):,}; "
            f"spam share {y.mean():.1%}; train/test = {1 - args.test_size:.0%}/{args.test_size:.0%}._", "",
            f"**Cross-validation on the training set ({args.cv}-fold, spam = positive class)**", "",
            df_to_md(cv_table.drop(columns="pr_auc_std").rename(columns={
                "model": "Model", "precision": "Precision", "recall": "Recall", "f1": "F1",
                "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"})), "",
            f"Best model by PR-AUC: **{best_name}**.", "",
            "**Held-out test set (default `predict()`)**", "",
            df_to_md(test_table.rename(columns={
                "model": "Model", "precision": "Precision", "recall": "Recall", "f1": "F1",
                "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"})), "",
            f"**Operating point.** Requirement: at least {args.target_precision:.0%} of blocked "
            f"messages must really be spam. Threshold {thr:.3f} (chosen on out-of-fold training "
            "scores"
            + ("" if reached else "; **the target was not reachable, so the max-precision point was used**")
            + f") gives on the test set: **{n_fp}** legitimate message(s) blocked and **{n_fn}** spam message(s) missed.", "",
            df_to_md(ops, ".3f"), "",
            "![Precision-recall curves](reports/figures/pr_curves.png)", "",
            "| Confusion matrix | Message length |", "| --- | --- |",
            "| ![Confusion matrix](reports/figures/confusion_matrix.png) | "
            "![Length distribution](reports/figures/length_distribution.png) |", "",
            "**What does the model look at?** Largest weights of a word-level logistic regression:", "",
            "![Top terms](reports/figures/top_terms.png)", "",
            "All misclassified test messages are saved to `reports/errors.csv` for error analysis.",
        ]
        update_block(ROOT / "README.md", "RESULTS", "\n".join(md))
        print("README.md updated (RESULTS block).")
    else:
        print(f"Demo run finished. Artefacts in {out} (README not modified).")
    return summary


if __name__ == "__main__":
    main()
