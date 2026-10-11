"""Classical TF-IDF baselines: Logistic Regression, SVM, Random Forest, XGBoost."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

log = logging.getLogger(__name__)


AVAILABLE_BASELINES = ["logreg", "svm", "rf", "xgboost"]


@dataclass
class TfidfBaselineResult:
    model_name: str
    vectorizer: TfidfVectorizer
    classifier: object
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]


def _build_classifier(name: str, *, seed: int = 42, class_weight: Optional[str] = "balanced"):
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=seed
        )
    if name == "svm":


        return LinearSVC(class_weight=class_weight, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight=class_weight, random_state=seed
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("xgboost is required for the 'xgboost' baseline") from e


        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown baseline {name!r}. Choose from {AVAILABLE_BASELINES}")


def train_tfidf_baseline(
    name: str,
    *,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    test_texts: Sequence[str],
    seed: int = 42,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_features: Optional[int] = 50000,
    class_weight: Optional[str] = "balanced",
) -> TfidfBaselineResult:
    """Train a TF-IDF + classical classifier and predict on ``test_texts``."""
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    clf = _build_classifier(name, seed=seed, class_weight=class_weight)
    clf.fit(X_train, train_labels)
    y_pred = np.asarray(clf.predict(X_test))
    y_proba: Optional[np.ndarray]
    if hasattr(clf, "predict_proba"):
        y_proba = np.asarray(clf.predict_proba(X_test))
    elif hasattr(clf, "decision_function"):

        scores = np.asarray(clf.decision_function(X_test))
        if scores.ndim == 1:

            from scipy.special import expit

            p = expit(scores)
            y_proba = np.stack([1.0 - p, p], axis=1)
        else:
            from scipy.special import softmax

            y_proba = softmax(scores, axis=1)
    else:
        y_proba = None
    return TfidfBaselineResult(
        model_name=name,
        vectorizer=vec,
        classifier=clf,
        y_pred=y_pred,
        y_proba=y_proba,
    )
