"""
Étape 6 — Entraînement du modèle final et export en .pkl.

Choisit la meilleure config de la baseline Text (TF-IDF + Logistic
Regression, cf. baseline_text.py), la ré-entraîne sur train+val, l'évalue
sur test, et sauvegarde :
  - models/modele.pkl         (pipeline sklearn complet : vectorizer + modèle)
  - models/modele_metadata.json (métriques, mapping des labels, date, config)

Usage:
    python src/train_final_model.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SPLITS_DIR, MODELS_DIR, RANDOM_SEED


def main():
    train = pd.read_csv(SPLITS_DIR / "train.csv")
    val = pd.read_csv(SPLITS_DIR / "val.csv")
    test = pd.read_csv(SPLITS_DIR / "test.csv")

    # Entraînement final sur train + val (le test reste intouché)
    train_full = pd.concat([train, val], ignore_index=True)
    X_train = train_full["transcript_clean"].fillna("")
    y_train = train_full["label"]
    X_test = test["transcript_clean"].fillna("")
    y_test = test["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_score)
    except ValueError:
        auc = None

    print("=== Modèle final (TF-IDF + Logistic Regression), entraîné sur train+val ===")
    print(classification_report(y_test, y_pred, target_names=["legitimate", "scam"]))

    model_path = MODELS_DIR / "modele.pkl"
    joblib.dump(pipeline, model_path)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "sklearn Pipeline: TfidfVectorizer(1,2-gram, max_features=20000) + LogisticRegression",
        "task": "Vishing detection — baseline Text seul (dataset ASLC-448 / arabic_scam_dataset_complete.xlsx, 448 conversations, 9 dialectes arabes)",
        "label_mapping": {"0": "legitimate", "1": "scam"},
        "n_train_val": len(train_full),
        "n_test": len(test),
        "test_metrics": {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        },
        "random_seed": RANDOM_SEED,
        "warning": (
            "Score proche de 1.0 attendu sur ce dataset synthétique à vocabulaire "
            "très marqué par catégorie — à documenter comme limite "
            "(signal lexical facile) dans le rapport, section 'Limites'."
        ),
    }
    meta_path = MODELS_DIR / "modele_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"\n-> Modèle sauvegardé : {model_path}")
    print(f"-> Métadonnées        : {meta_path}")
    print("\nPour réutiliser le modèle :")
    print("    import joblib")
    print("    pipeline = joblib.load('models/modele.pkl')")
    print("    pipeline.predict(['texte de conversation ici'])")


if __name__ == "__main__":
    main()
