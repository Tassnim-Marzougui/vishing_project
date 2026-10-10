"""
Étape 3 — Splits train/val/test (cahier des charges section 6.2 / 8.3).

Séparation STRICTE par conversation_id (pas par ligne), stratifiée par
dialecte + label dans la mesure du possible, pour éviter toute fuite de
locuteur/scénario entre les ensembles.

Entrée  : data/processed/metadata_clean.csv
Sortie  : data/splits/{train,val,test}.csv

Usage:
    python src/make_splits.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, SPLITS_DIR, RANDOM_SEED, TEST_SIZE, VAL_SIZE


def main():
    in_path = PROCESSED_DIR / "metadata_clean.csv"
    if not in_path.exists():
        print("[!] Lance d'abord `python src/preprocess.py`.")
        return

    df = pd.read_csv(in_path)

    # Une ligne par conversation_id, avec le label majoritaire de cette conversation
    # (permet la stratification par classe au niveau conversation, pas ligne).
    conv_labels = df.groupby("conversation_id")["label"].agg(lambda s: s.mode().iat[0])
    conv_dialects = df.groupby("conversation_id")["dialect"].agg(lambda s: s.mode().iat[0])
    conv_df = pd.DataFrame({"label": conv_labels, "dialect": conv_dialects}).reset_index()

    # clé de stratification combinée (dialecte + label) ; fallback sur label seul
    # si une strate est trop petite (train_test_split exige >= 2 par strate).
    strat_key = conv_df["dialect"].astype(str) + "_" + conv_df["label"].astype(str)
    strat = strat_key if strat_key.value_counts().min() >= 2 else conv_df["label"]

    train_ids, temp_ids = train_test_split(
        conv_df["conversation_id"],
        test_size=TEST_SIZE + VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=strat,
    )

    temp_df = conv_df[conv_df["conversation_id"].isin(temp_ids)]
    temp_strat_key = temp_df["dialect"].astype(str) + "_" + temp_df["label"].astype(str)
    temp_strat = temp_strat_key if temp_strat_key.value_counts().min() >= 2 else temp_df["label"]

    rel_test_size = TEST_SIZE / (TEST_SIZE + VAL_SIZE)
    val_ids, test_ids = train_test_split(
        temp_df["conversation_id"],
        test_size=rel_test_size,
        random_state=RANDOM_SEED,
        stratify=temp_strat,
    )

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    for name, ids in splits.items():
        split_df = df[df["conversation_id"].isin(ids)]
        out_path = SPLITS_DIR / f"{name}.csv"
        split_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"{name:5s}: {len(split_df):5d} lignes | {len(ids):4d} conversations "
              f"| positif={split_df['label'].mean():.2%}  -> {out_path}")

    # vérification stricte : aucune conversation_id partagée entre splits
    overlap_train_val = set(train_ids) & set(val_ids)
    overlap_train_test = set(train_ids) & set(test_ids)
    overlap_val_test = set(val_ids) & set(test_ids)
    assert not (overlap_train_val or overlap_train_test or overlap_val_test), \
        "Fuite de données détectée entre splits !"
    print("\n[OK] Aucune fuite de conversation_id entre train/val/test.")


if __name__ == "__main__":
    main()
