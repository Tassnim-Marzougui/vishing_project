# Détection du Vishing — Sprint 2 (Data Preparation + Baselines)

Dataset cible : **ASLC-448** (Arabic Scam/Legitimate Call Conversation Dataset),
Mendeley Data, DOI 10.17632/p384bgyzz3.4 — 448 conversations, 9 dialectes arabes
(MSA, Egyptian, Gulf, Jordanian, Saudi, Yemeni, Sudanese, Iraqi, Syrian), audio
synthétisé (TTS), labels scam / legitimate.

## 0. Avant de commencer

1. Télécharge le dataset depuis Mendeley :
   https://data.mendeley.com/datasets/p384bgyzz3/4
2. Dépose le contenu (zip extrait, CSV/Excel de métadonnées, dossiers audio) dans
   `data/raw/`.
3. Lance `python src/explore_dataset.py` pour voir la structure réelle (colonnes,
   nombre de fichiers, distribution des classes/dialectes). Ce script ne suppose
   rien — il inspecte ce qu'il trouve et affiche un rapport.
4. Ajuste les noms de colonnes dans `src/config.py` si besoin (le script
   `explore_dataset.py` te dira quoi corriger).

## 1. Pipeline Sprint 2 (dans l'ordre)

| Étape | Script | Sortie |
|---|---|---|
| 1. Exploration | `src/explore_dataset.py` | rapport console + `results/dataset_report.json` |
| 2. Prétraitement | `src/preprocess.py` | `data/processed/metadata_clean.csv` |
| 3. Splits | `src/make_splits.py` | `data/splits/{train,val,test}.csv` |
| 4. Baseline Text | `src/baseline_text.py` | `results/baseline_text_metrics.json` |
| 5. Baseline Audio | `src/baseline_audio.py` | `results/baseline_audio_metrics.json` |

```bash
pip install -r requirements.txt
python src/explore_dataset.py
python src/preprocess.py
python src/make_splits.py
python src/baseline_text.py
python src/baseline_audio.py
```

## 2. Points d'attention (cahier des charges)

- **Pas de fuite de données** : les splits sont faits par conversation/locuteur
  (jamais par ligne de transcript), et stratifiés par dialecte + label.
- **Harmonisation des schémas d'annotation** : `preprocess.py` normalise les
  valeurs de label (`scam`/`vishing`/`fraud` → `1`, `legitimate`/`safe` → `0`)
  et les noms de dialecte.
- **Reproductibilité** : seed fixe (`RANDOM_SEED` dans `config.py`).
