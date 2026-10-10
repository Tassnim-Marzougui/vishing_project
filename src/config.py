"""
Configuration centrale du projet — Sprint 2.

À AJUSTER une fois que `explore_dataset.py` a montré la structure réelle
du dataset ASLC-448 (noms de colonnes, format des fichiers, etc.).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

for d in (PROCESSED_DIR, SPLITS_DIR, RESULTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# --- Colonnes attendues dans le(s) fichier(s) de métadonnées ---
# Ces noms sont des CANDIDATS probables (variantes courantes dans ce type de
# dataset). explore_dataset.py essaie de les détecter automatiquement parmi
# les colonnes réelles ; mets à jour cette liste si la détection échoue.

CANDIDATE_LABEL_COLS = ["label_binary", "label", "class", "category", "type", "is_scam", "scam_label"]
CANDIDATE_DIALECT_COLS = ["dialect", "accent", "region", "arabic_dialect"]
CANDIDATE_TRANSCRIPT_COLS = ["full_conversation", "transcript", "text", "conversation", "dialogue", "utterance"]
CANDIDATE_FILENAME_COLS = ["filename", "file", "audio_file", "file_name", "path", "audio_path"]
CANDIDATE_SPEAKER_COLS = ["conversation_id", "speaker", "speaker_id", "caller_id", "call_id"]

# --- Harmonisation des labels ---
# Toute valeur (en minuscules) qui matche une de ces listes est mappée vers 0/1.
POSITIVE_LABEL_VALUES = {"scam", "vishing", "fraud", "fraudulent", "phishing", "1", "malicious"}
NEGATIVE_LABEL_VALUES = {"legitimate", "legit", "safe", "not_scam", "non-scam", "non_scam", "0", "normal"}

# --- Colonnes de features additionnelles (arabic_scam_dataset_complete.xlsx) ---
# Scores déjà calculés dans le dataset, utilisables pour un modèle enrichi.
EXTRA_NUMERIC_FEATURE_COLS = [
    "urgency_score", "sensitive_info_requests", "financial_pressure_score",
    "threat_score", "impersonation_score", "conversation_length", "word_count",
]
CATEGORY_COL = "category"  # sous-type (bank_fraud, prize_lottery, ...) — info, pas le label

# --- Dialectes connus (ASLC-448) ---
KNOWN_DIALECTS = [
    "MSA", "Egyptian", "Gulf", "Jordanian", "Saudi",
    "Yemeni", "Sudanese", "Iraqi", "Syrian",
]

# --- Audio ---
TARGET_SAMPLE_RATE = 16000
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# --- Splits ---
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # relatif au total, pris sur le reste après le test split
