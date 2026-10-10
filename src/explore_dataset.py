"""
Étape 1 — Exploration du dataset ASLC-448 brut.

Ne suppose rien sur la structure : inspecte data/raw/, liste les fichiers,
charge tout CSV/Excel trouvé, et essaie de détecter automatiquement les
colonnes label / dialecte / transcript / nom de fichier / locuteur.

Usage:
    python src/explore_dataset.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_DIR, RESULTS_DIR, AUDIO_EXTENSIONS,
    CANDIDATE_LABEL_COLS, CANDIDATE_DIALECT_COLS,
    CANDIDATE_TRANSCRIPT_COLS, CANDIDATE_FILENAME_COLS, CANDIDATE_SPEAKER_COLS,
)


def find_tabular_files(root: Path):
    exts = {".csv", ".xlsx", ".xls", ".json", ".tsv"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def find_audio_files(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]


def load_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError(f"Format non supporté: {path}")


def detect_column(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # recherche partielle (contient)
    for cand in candidates:
        for c_lower, c_orig in cols_lower.items():
            if cand in c_lower:
                return c_orig
    return None


def main():
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(f"[!] {RAW_DIR} est vide.")
        print("    -> Dépose le dataset ASLC-448 (zip extrait) dans ce dossier,")
        print("       puis relance ce script.")
        return

    report = {"raw_dir": str(RAW_DIR)}

    tabular_files = find_tabular_files(RAW_DIR)
    audio_files = find_audio_files(RAW_DIR)

    print(f"Fichiers tabulaires trouvés : {len(tabular_files)}")
    for f in tabular_files:
        print(f"  - {f.relative_to(RAW_DIR)}")
    print(f"Fichiers audio trouvés : {len(audio_files)}")
    if audio_files:
        ext_counts = Counter(f.suffix.lower() for f in audio_files)
        print(f"  Extensions : {dict(ext_counts)}")

    report["n_tabular_files"] = len(tabular_files)
    report["n_audio_files"] = len(audio_files)
    report["audio_extensions"] = dict(Counter(f.suffix.lower() for f in audio_files))

    if not tabular_files:
        print("\n[!] Aucun fichier de métadonnées (csv/xlsx/json) trouvé.")
        print("    Vérifie que le dataset a bien été extrait dans data/raw/.")
        RESULTS_DIR.joinpath("dataset_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        return

    # On inspecte le premier fichier tabulaire (le plus probable = métadonnées principales)
    main_table_path = max(tabular_files, key=lambda p: p.stat().st_size)
    print(f"\n--- Inspection du plus gros fichier tabulaire : {main_table_path.name} ---")
    df = load_tabular(main_table_path)
    print(f"Shape: {df.shape}")
    print(f"Colonnes: {list(df.columns)}")
    print(df.head(5).to_string())

    label_col = detect_column(df.columns, CANDIDATE_LABEL_COLS)
    dialect_col = detect_column(df.columns, CANDIDATE_DIALECT_COLS)
    transcript_col = detect_column(df.columns, CANDIDATE_TRANSCRIPT_COLS)
    filename_col = detect_column(df.columns, CANDIDATE_FILENAME_COLS)
    speaker_col = detect_column(df.columns, CANDIDATE_SPEAKER_COLS)

    print("\n--- Détection automatique des colonnes ---")
    print(f"  Label (scam/legitimate) : {label_col!r}")
    print(f"  Dialecte                : {dialect_col!r}")
    print(f"  Transcript               : {transcript_col!r}")
    print(f"  Nom de fichier audio     : {filename_col!r}")
    print(f"  Locuteur / ID conv.      : {speaker_col!r}")

    if label_col:
        print(f"\nDistribution des labels ({label_col}):")
        print(df[label_col].value_counts(dropna=False))
    if dialect_col:
        print(f"\nDistribution des dialectes ({dialect_col}):")
        print(df[dialect_col].value_counts(dropna=False))

    missing = [name for name, val in [
        ("label", label_col), ("transcript", transcript_col), ("filename", filename_col)
    ] if val is None]
    if missing:
        print(f"\n[!] Colonnes non détectées automatiquement : {missing}")
        print("    -> Ouvre le fichier et mets à jour src/config.py "
              "(CANDIDATE_*_COLS) avec les vrais noms, puis relance.")

    report.update({
        "main_table_file": main_table_path.name,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "detected_label_col": label_col,
        "detected_dialect_col": dialect_col,
        "detected_transcript_col": transcript_col,
        "detected_filename_col": filename_col,
        "detected_speaker_col": speaker_col,
    })
    RESULTS_DIR.joinpath("dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str)
    )
    print(f"\nRapport sauvegardé -> {RESULTS_DIR / 'dataset_report.json'}")


if __name__ == "__main__":
    main()
