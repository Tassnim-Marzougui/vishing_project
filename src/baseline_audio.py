"""
Étape 5 — Baseline Audio seul (cahier des charges section 6.3 / 8.3).

Features acoustiques classiques (MFCC + dérivées, pitch, énergie) agrégées
par statistiques (moyenne/écart-type), + Random Forest / Gradient Boosting.
Choix volontaire d'une baseline légère (section 9 "Risques" : "Coût de
calcul -> baselines légères puis complexité croissante") avant de passer à
des embeddings de parole pré-entraînés en Sprint 3.

Entrée  : data/splits/{train,val,test}.csv (colonne audio_file) + fichiers
          audio réels sous data/raw/.
Sortie  : results/baseline_audio_metrics.json

Usage:
    python src/baseline_audio.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SPLITS_DIR, RESULTS_DIR, RAW_DIR, RANDOM_SEED, TARGET_SAMPLE_RATE

try:
    import librosa
except ImportError:
    librosa = None

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report,
)


def find_audio_path(filename: str) -> Path | None:
    if not isinstance(filename, str) or not filename:
        return None
    direct = RAW_DIR / filename
    if direct.exists():
        return direct
    matches = list(RAW_DIR.rglob(Path(filename).name))
    return matches[0] if matches else None


def extract_features(path: Path) -> np.ndarray | None:
    try:
        y, sr = librosa.load(path, sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"[!] Impossible de charger {path}: {e}")
        return None
    if len(y) < sr * 0.2:  # moins de 200ms -> trop court
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[np.isfinite(f0)]
        pitch_mean = float(np.mean(f0)) if len(f0) else 0.0
        pitch_std = float(np.std(f0)) if len(f0) else 0.0
    except Exception:
        pitch_mean, pitch_std = 0.0, 0.0

    feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
        zcr.mean(axis=1), zcr.std(axis=1),
        rmse.mean(axis=1), rmse.std(axis=1),
        centroid.mean(axis=1), centroid.std(axis=1),
        [pitch_mean, pitch_std],
    ])
    return feats


def build_feature_matrix(df: pd.DataFrame):
    X, y, kept_idx = [], [], []
    for i, row in df.iterrows():
        path = find_audio_path(row.get("audio_file"))
        if path is None:
            continue
        feats = extract_features(path)
        if feats is None:
            continue
        X.append(feats)
        y.append(row["label"])
        kept_idx.append(i)
    if not X:
        return None, None
    return np.vstack(X), np.array(y)


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
    if librosa is None:
        print("[!] librosa n'est pas installé. Lance : pip install librosa soundfile")
        return

    train = pd.read_csv(SPLITS_DIR / "train.csv")
    val = pd.read_csv(SPLITS_DIR / "val.csv")
    test = pd.read_csv(SPLITS_DIR / "test.csv")

    if "audio_file" not in train.columns or train["audio_file"].isna().all():
        print("[!] Pas de colonne audio_file exploitable dans les splits.")
        print("    Vérifie CANDIDATE_FILENAME_COLS dans config.py et relance preprocess.py.")
        return

    print("Extraction des features audio (train)...")
    Xtr, ytr = build_feature_matrix(train)
    print("Extraction des features audio (val)...")
    Xva, yva = build_feature_matrix(val)
    print("Extraction des features audio (test)...")
    Xte, yte = build_feature_matrix(test)

    if Xtr is None or Xte is None:
        print("[!] Aucun fichier audio n'a pu être chargé depuis data/raw/.")
        print("    Vérifie que les fichiers audio du dataset sont bien présents.")
        return

    print(f"Train: {Xtr.shape}, Val: {Xva.shape if Xva is not None else None}, Test: {Xte.shape}")

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    Xva_s = scaler.transform(Xva) if Xva is not None and len(Xva) else None

    results = {}

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(Xtr_s, ytr)
    test_pred = rf.predict(Xte_s)
    test_score = rf.predict_proba(Xte_s)[:, 1]
    results["random_forest"] = {"test": evaluate(yte, test_pred, test_score)}
    if Xva_s is not None and len(Xva_s):
        results["random_forest"]["val"] = evaluate(yva, rf.predict(Xva_s))
    print("=== Random Forest (features acoustiques) ===")
    print(classification_report(yte, test_pred, target_names=["legitimate", "scam"]))

    gbt = GradientBoostingClassifier(random_state=RANDOM_SEED)
    gbt.fit(Xtr_s, ytr)
    test_pred_gbt = gbt.predict(Xte_s)
    test_score_gbt = gbt.predict_proba(Xte_s)[:, 1]
    results["gradient_boosting"] = {"test": evaluate(yte, test_pred_gbt, test_score_gbt)}
    print("\n=== Gradient Boosting (features acoustiques) ===")
    print(classification_report(yte, test_pred_gbt, target_names=["legitimate", "scam"]))

    out_path = RESULTS_DIR / "baseline_audio_metrics.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
