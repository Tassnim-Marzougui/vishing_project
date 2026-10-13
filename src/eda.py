"""
EDA complète — dataset ASLC-448 (arabic_scam_dataset_complete.xlsx)

Usage (depuis la racine vishing_project) :
    python src/eda.py

Sorties :
    results/eda/*.png          (graphiques)
    results/eda/eda_report.txt (tout ce qui est affiché dans le terminal)
"""

import os
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Pour afficher correctement l'arabe dans les graphiques (optionnel)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    def ar(text):
        return get_display(arabic_reshaper.reshape(str(text)))
except ImportError:
    def ar(text):
        return str(text)
    print("[info] pip install arabic-reshaper python-bidi pour avoir l'arabe lisible dans les graphiques\n")

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "raw" / "arabic_scam_dataset_complete.xlsx"
OUT = ROOT / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


# Tout ce qui est affiché est aussi écrit dans eda_report.txt
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)

    def flush(self):
        for f in self.files:
            f.flush()


report_file = open(OUT / "eda_report.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, report_file)


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=130)
    plt.close()


df = pd.read_excel(PATH)

# ---------------------------------------------------------------
section("1. Qualité des données")
print("Lignes :", len(df))
print("Colonnes :", list(df.columns))
print("Valeurs manquantes (total) :", int(df.isnull().sum().sum()))
print("Doublons lignes entières :", df.duplicated().sum())
print("Doublons full_conversation :", df["full_conversation"].duplicated().sum())
print("Doublons caller_turn_1 :", df["caller_turn_1"].duplicated().sum())
print("Textes vides :", int((df["full_conversation"].str.strip() == "").sum()))
print("conversation_id uniques :", df["conversation_id"].nunique())

# ---------------------------------------------------------------
section("2. Répartition des classes, dialectes, catégories")
print(df["label"].value_counts())
print("\nPourcentage :")
print((df["label"].value_counts(normalize=True) * 100).round(1))
print("\nDialecte x Label :")
print(pd.crosstab(df["dialect"], df["label"]))
print("\nCatégorie x Label :")
print(pd.crosstab(df["category"], df["label"]))

df["label"].value_counts().plot(kind="bar", title="Répartition scam / not_scam", rot=0)
save("01_label.png")

pd.crosstab(df["dialect"], df["label"]).plot(kind="bar", title="Dialecte x Label")
save("02_dialect_label.png")

pd.crosstab(df["category"], df["label"]).plot(
    kind="barh", stacked=True, figsize=(8, 8), title="Catégorie x Label")
save("03_category_label.png")

# ---------------------------------------------------------------
section("3. Longueur et scores par label")
num_cols = ["word_count", "conversation_length", "urgency_score",
            "sensitive_info_requests", "financial_pressure_score",
            "threat_score", "impersonation_score"]
print(df.groupby("label")[num_cols].mean().round(2).T)
print("\nDescription globale :")
print(df[num_cols].describe().round(2).T)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col in zip(axes, ["word_count", "conversation_length"]):
    for lab, color in [("not_scam", "tab:blue"), ("scam", "tab:red")]:
        ax.hist(df.loc[df["label"] == lab, col], bins=25, alpha=0.6, label=lab, color=color)
    ax.set_title(f"Distribution de {col}")
    ax.legend()
save("04_histogrammes_longueur.png")

df.groupby("label")[num_cols[2:]].mean().T.plot(kind="bar", figsize=(9, 4),
                                               title="Scores moyens par label", rot=30)
save("05_scores_moyens.png")

# ---------------------------------------------------------------
section("4. Corrélation entre variables numériques et label")
corr_df = df[num_cols + ["label_binary"]].corr()
print(corr_df["label_binary"].drop("label_binary").sort_values(ascending=False).round(3))

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_df)))
ax.set_yticks(range(len(corr_df)))
ax.set_xticklabels(corr_df.columns, rotation=60, ha="right")
ax.set_yticklabels(corr_df.columns)
for i in range(len(corr_df)):
    for j in range(len(corr_df)):
        ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im)
ax.set_title("Corrélation")
save("06_correlation.png")

# ---------------------------------------------------------------
section("5. Mots les plus fréquents par classe")
STOP = {"المتصل", "المستقبل", "أنا", "على", "هذا", "هذه", "التي", "الذي", "من", "في", "عن", "إلى", "مع"}


def tokenize(text):
    text = re.sub(r"[ً-ْـ]", "", text)
    return [w for w in re.findall(r"[؀-ۿ]+", text) if len(w) > 2 and w not in STOP]


top_words = {}
for lab in ["scam", "not_scam"]:
    words = []
    for t in df.loc[df["label"] == lab, "full_conversation"]:
        words += tokenize(t)
    top_words[lab] = Counter(words).most_common(15)
    print(lab, top_words[lab][:10])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, lab, color in zip(axes, ["not_scam", "scam"], ["tab:blue", "tab:red"]):
    labels = [ar(w) for w, _ in top_words[lab]][::-1]
    values = [c for _, c in top_words[lab]][::-1]
    ax.barh(labels, values, color=color)
    ax.set_title(f"Top mots — {lab}")
save("07_top_mots.png")

# ---------------------------------------------------------------
section("6. Mots les plus discriminants (coefficients Logistic Regression)")
vec = TfidfVectorizer(ngram_range=(1, 1), min_df=3, tokenizer=tokenize, lowercase=False,
                      token_pattern=None)
Xw = vec.fit_transform(df["full_conversation"])
lr = LogisticRegression(max_iter=1000).fit(Xw, df["label_binary"])
vocab = np.array(vec.get_feature_names_out())
order = np.argsort(lr.coef_[0])
print("Mots -> not_scam :", list(vocab[order[:12]]))
print("Mots -> scam     :", list(vocab[order[-12:][::-1]]))

top_n = 12
idx = np.concatenate([order[:top_n], order[-top_n:]])
fig, ax = plt.subplots(figsize=(7, 7))
ax.barh([ar(w) for w in vocab[idx]], lr.coef_[0][idx],
        color=["tab:blue"] * top_n + ["tab:red"] * top_n)
ax.set_title("Mots discriminants (bleu = not_scam, rouge = scam)")
save("08_mots_discriminants.png")

# ---------------------------------------------------------------
section("7. Quasi-doublons (fuite possible entre train et test)")
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(df["full_conversation"])
sim = cosine_similarity(tfidf)
np.fill_diagonal(sim, 0)
max_sim = sim.max(axis=1)
nearest = sim.argmax(axis=1)

for th in (0.5, 0.7, 0.8, 0.9):
    print(f"Conversations avec un voisin similaire > {th} : {(max_sim > th).sum()} / {len(df)}")

same_label = (df["label"].values == df["label"].values[nearest]).mean()
print(f"\nLe voisin le plus proche a le MÊME label dans {same_label:.1%} des cas")
same_cat = (df["category"].values == df["category"].values[nearest]).mean()
print(f"Le voisin le plus proche a la MÊME catégorie dans {same_cat:.1%} des cas")

plt.hist(max_sim, bins=30, color="tab:purple")
plt.title("Similarité max avec une autre conversation")
plt.xlabel("cosine similarity")
save("09_quasi_doublons.png")

# ---------------------------------------------------------------
section("8. Longueur des tours de parole")
turn_cols = ["caller_turn_1", "receiver_turn_1", "caller_turn_2", "receiver_turn_2", "caller_turn_3"]
turn_len = pd.DataFrame({c: df[c].astype(str).str.split().str.len() for c in turn_cols})
turn_len["label"] = df["label"]
print(turn_len.groupby("label").mean().round(1).T)

# ---------------------------------------------------------------
section("9. Résumé : ce qu'il faut retenir")
print(f"- Classes : {df['label'].value_counts().to_dict()}")
print(f"- Les scams sont plus longs ({df.loc[df.label=='scam','word_count'].mean():.0f} mots "
      f"vs {df.loc[df.label=='not_scam','word_count'].mean():.0f}).")
print(f"- Aucune catégorie n'est partagée entre scam et not_scam : "
      f"{(pd.crosstab(df['category'], df['label']) > 0).sum(axis=1).max() == 1}")
print(f"- Voisin le plus proche de même catégorie : {same_cat:.1%} -> le modèle reconnaît surtout "
      f"le SUJET (banque, livraison...), pas seulement la fraude.")
print("- Limite à écrire dans le rapport : dataset synthétique, vocabulaire très marqué -> score ~1.00.")

print(f"\nGraphiques + rapport dans : {os.path.abspath(OUT)}")
sys.stdout = sys.__stdout__
report_file.close()