"""Model zoo: every model is a full text -> prediction Pipeline."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from .text import meta_features, normalize_text


def word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2),
                           min_df=2, sublinear_tf=True)


def char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(preprocessor=normalize_text, analyzer="char_wb",
                           ngram_range=(2, 5), min_df=3, sublinear_tf=True)


def meta_block() -> Pipeline:
    return Pipeline([("stats", FunctionTransformer(meta_features)),
                     ("scale", StandardScaler())])


def get_models(seed: int) -> dict[str, BaseEstimator]:
    return {
        "Baseline (majority class)": Pipeline([
            ("meta", meta_block()), ("clf", DummyClassifier(strategy="prior"))]),
        "Hand-crafted features + LR": Pipeline([
            ("meta", meta_block()), ("clf", LogisticRegression(max_iter=1000))]),
        "TF-IDF + Naive Bayes": Pipeline([
            ("tfidf", word_tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "TF-IDF + Logistic regression": Pipeline([
            ("tfidf", word_tfidf()),
            ("clf", LogisticRegression(C=10, max_iter=2000))]),
        "TF-IDF (word+char) + Linear SVM": Pipeline([
            ("features", FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])),
            ("clf", LinearSVC(C=0.5, dual="auto", random_state=seed))]),
        "TF-IDF + hand-crafted + LR": Pipeline([
            ("features", FeatureUnion([("word", word_tfidf()), ("char", char_tfidf()),
                                       ("meta", meta_block())])),
            ("clf", LogisticRegression(C=10, max_iter=3000))]),
    }


def get_scores(model, texts) -> np.ndarray:
    """Spam score: probability when available, otherwise the SVM margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)
