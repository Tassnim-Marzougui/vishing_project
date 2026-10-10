"""
Étape 4 — Baseline Text seul (cahier des charges section 6.3 / 8.3).

TF-IDF + Régression Logistique et Linear SVM sur les transcripts nettoyés.
Rapporte Accuracy, Precision, Recall, F1-score (section 6.5).

Entrée  : data/splits/{train,val,test}.csv
Sortie  : results/baseline_text_metrics.json

Usage:
    python src/baseline_text.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SPLITS_DIR, RESULTS_DIR, RANDOM_SEED


def load_split(name):
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} manquant — lance `python src/make_splits.py` d'abord.")
    return pd.read_csv(path)


def evaluate(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    if y_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            metrics["roc_auc"] = None
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    X_train, y_train = train["transcript_clean"].fillna(""), train["label"]
    X_val, y_val = val["transcript_clean"].fillna(""), val["label"]
    X_test, y_test = test["transcript_clean"].fillna(""), test["label"]

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)
    Xte = vectorizer.transform(X_test)

    results = {}

    # --- Logistic Regression ---
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    logreg.fit(Xtr, y_train)
    val_pred = logreg.predict(Xva)
    val_score = logreg.predict_proba(Xva)[:, 1]
    test_pred = logreg.predict(Xte)
    test_score = logreg.predict_proba(Xte)[:, 1]
    results["logistic_regression"] = {
        "val": evaluate(y_val, val_pred, val_score),
        "test": evaluate(y_test, test_pred, test_score),
    }
    print("=== Logistic Regression (TF-IDF) ===")
    print(classification_report(y_test, test_pred, target_names=["legitimate", "scam"]))

    # --- Linear SVM ---
    svm = LinearSVC(class_weight="balanced", random_state=RANDOM_SEED)
    svm.fit(Xtr, y_train)
    val_pred_svm = svm.predict(Xva)
    test_pred_svm = svm.predict(Xte)
    test_score_svm = svm.decision_function(Xte)
    results["linear_svm"] = {
        "val": evaluate(y_val, val_pred_svm),
        "test": evaluate(y_test, test_pred_svm, test_score_svm),
    }
    print("\n=== Linear SVM (TF-IDF) ===")
    print(classification_report(y_test, test_pred_svm, target_names=["legitimate", "scam"]))

    out_path = RESULTS_DIR / "baseline_text_metrics.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
