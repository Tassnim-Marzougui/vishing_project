
"""
Etape 2 - Pretraitement du dataset Vishing Detection.

- Nettoyage des transcriptions arabes.
- Harmonisation des labels (scam=1, legitimate=0).
- Harmonisation des dialectes.
- Association automatique des conversations aux fichiers WAV.
- Generation de data/processed/metadata_clean.csv.

Usage :
    python src/preprocess.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    POSITIVE_LABEL_VALUES,
    NEGATIVE_LABEL_VALUES,
    KNOWN_DIALECTS,
)


# ============================================================
# 1. Nettoyage du texte arabe
# ============================================================

ARABIC_DIACRITICS = re.compile(
    r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]"
)

REPEATED_SPACES = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    t = text.strip()
    t = URL_RE.sub(" ", t)
    t = ARABIC_DIACRITICS.sub("", t)
    t = t.replace("\u0640", "")
    t = REPEATED_SPACES.sub(" ", t).strip()

    return t


# ============================================================
# 2. Harmonisation des labels
# ============================================================

def harmonize_label(value) -> int | None:
    if pd.isna(value):
        return None

    v = str(value).strip().lower()

    if v in POSITIVE_LABEL_VALUES:
        return 1

    if v in NEGATIVE_LABEL_VALUES:
        return 0

    try:
        iv = int(float(v))
        if iv in (0, 1):
            return iv
    except (ValueError, OverflowError):
        pass

    return None


# ============================================================
# 3. Harmonisation des dialectes
# ============================================================

def harmonize_dialect(value) -> str:
    if not isinstance(value, str) or not value.strip():
        return "Unknown"

    v = value.strip().lower()

    for known in KNOWN_DIALECTS:
        if known.lower() == v or known.lower() in v:
            return known

    return value.strip().title()


# ============================================================
# 4. Association automatique des fichiers audio
# ============================================================

def build_audio_mapping():
    """
    Exemple :
        CONV_0001 -> audio/CONV_0001.wav

    Les chemins sont relatifs a data/raw.
    """

    audio_mapping = {}

    for path in RAW_DIR.rglob("*.wav"):
        conversation_id = path.stem.strip().upper()

        audio_mapping[conversation_id] = (
            path.relative_to(RAW_DIR).as_posix()
        )

    return audio_mapping


# ============================================================
# 5. Pipeline principal
# ============================================================

def main():

    print("\n=== PRETRAITEMENT VISHING DATASET ===\n")

    # Charger le rapport d'exploration
    report_path = RESULTS_DIR / "dataset_report.json"

    if not report_path.exists():
        raise FileNotFoundError(
            "Rapport introuvable. Lance src/explore_dataset.py."
        )

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    # Identifier le fichier principal
    main_table = (
        RAW_DIR / report["main_table_file"]
        if report.get("main_table_file")
        else None
    )

    if main_table is None or not main_table.exists():

        filename = report.get("main_table_file")

        if filename:
            candidates = list(RAW_DIR.rglob(Path(filename).name))
        else:
            candidates = (
                list(RAW_DIR.rglob("*.xlsx"))
                + list(RAW_DIR.rglob("*.csv"))
            )

        main_table = candidates[0] if candidates else None

    if main_table is None:
        raise FileNotFoundError(
            "Fichier principal introuvable dans data/raw."
        )

    print(f"Dataset : {main_table.name}")

    # Charger les donnees
    suffix = main_table.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(main_table)

    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(main_table)

    elif suffix == ".json":
        df = pd.read_json(main_table)

    else:
        raise ValueError(
            f"Format non pris en charge : {suffix}"
        )

    print(f"Conversations initiales : {len(df)}")

    # Colonnes detectees pendant l'exploration
    label_col = report.get("detected_label_col")
    dialect_col = report.get("detected_dialect_col")
    transcript_col = report.get("detected_transcript_col")
    filename_col = report.get("detected_filename_col")
    speaker_col = report.get("detected_speaker_col")

    if not label_col or not transcript_col:
        raise ValueError(
            "Colonnes label ou transcript non detectees."
        )

    # Construire les metadonnees nettoyees
    out = pd.DataFrame(index=df.index)

    out["transcript_raw"] = df[transcript_col]

    out["transcript_clean"] = (
        out["transcript_raw"].apply(clean_text)
    )

    out["label"] = (
        df[label_col].apply(harmonize_label)
    )

    if dialect_col:
        out["dialect"] = (
            df[dialect_col].apply(harmonize_dialect)
        )
    else:
        out["dialect"] = "Unknown"

    # Identifiants des conversations
    if speaker_col:
        out["conversation_id"] = (
            df[speaker_col].astype(str).str.strip()
        )
    else:
        out["conversation_id"] = (
            pd.Series(
                range(len(df)),
                index=df.index
            ).astype(str)
        )

    # Associer les fichiers audio
    audio_mapping = build_audio_mapping()

    print(f"Fichiers WAV trouves : {len(audio_mapping)}")

    if filename_col:
        out["audio_file"] = df[filename_col]
    else:
        out["audio_file"] = None

    # Completer les chemins audio manquants
    missing_audio = (
        out["audio_file"].isna()
        | out["audio_file"].astype(str).str.strip().eq("")
    )

    out.loc[missing_audio, "audio_file"] = (
        out.loc[missing_audio, "conversation_id"]
        .str.upper()
        .map(audio_mapping)
    )

    # Verifier que les chemins pointent vers des fichiers
    def validate_audio_path(value):
        if pd.isna(value):
            return None

        path_str = str(value).strip()

        if not path_str:
            return None

        path = Path(path_str)

        if not path.is_absolute():
            path = RAW_DIR / path

        if not path.is_file():
            return None

        return path.relative_to(RAW_DIR).as_posix()

    out["audio_file"] = (
        out["audio_file"].apply(validate_audio_path)
    )

    # Nettoyer les lignes invalides
    n_before = len(out)
    n_unlabeled = out["label"].isna().sum()

    out = out.dropna(subset=["label"])
    out = out[
        out["transcript_clean"].str.len() > 0
    ].copy()

    out["label"] = out["label"].astype(int)

    n_after = len(out)

    # Statistiques audio
    n_audio = out["audio_file"].notna().sum()
    n_missing_audio = n_after - n_audio

    # Sauvegarder
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path = (
        PROCESSED_DIR / "metadata_clean.csv"
    )

    out.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Afficher le rapport
    print("\n=== RESULTATS DU PRETRAITEMENT ===")

    print(f"Lignes avant nettoyage : {n_before}")
    print(f"Labels invalides : {n_unlabeled}")
    print(f"Lignes finales : {n_after}")

    print(f"\nAudios associes : {n_audio}/{n_after}")
    print(f"Audios manquants : {n_missing_audio}")

    print("\nDistribution des classes :")
    print(out["label"].value_counts())

    print("\nDistribution des dialectes :")
    print(out["dialect"].value_counts())

    print(f"\nFichier sauvegarde : {out_path}")

    print("\n[OK] Pretraitement termine avec succes.")


if __name__ == "__main__":
    main()
