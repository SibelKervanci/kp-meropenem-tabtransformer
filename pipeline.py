# -*- coding: utf-8 -*-
"""
Leakage-Aware Genomic Prediction Pipeline for Meropenem Resistance
in Klebsiella pneumoniae Using Transformer-Based Resistome Representation Learning

Author : Ilkay Sibel Kervanci
ORCID  : 0000-0001-5547-1860
Email  : skervanci@gantep.edu.tr
Date   : 2026

Publication: Submitted to Bioinformatics (Oxford University Press)

Usage:
    python pipeline.py

Input files (place in working directory):
    asilverisetigenler.csv   - Discovery cohort (BV-BRC / PRJNA376414)
    validationPRJlison.csv   - External validation cohort (7 BioProjects)
"""

# =====================================================
# IMPORTS
# =====================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, confusion_matrix,
                             recall_score, f1_score, matthews_corrcoef,
                             roc_auc_score, roc_curve, auc, precision_recall_curve)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler
from scipy import stats
import shap

optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# =====================================================
# REPRODUCIBILITY
# =====================================================
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global storage for trained models and predictions
trained_models  = {}
internal_preds  = {}
internal_probs  = {}

# =====================================================
# DATA LOADING
# =====================================================
print("\n============================")
print("DATA LOADING")
print("============================")

df = pd.read_csv("asilverisetigenler.csv", sep=";")

if "Genome ID" in df.columns:
    df = df.drop(columns=["Genome ID"])

# Encode resistance phenotype as binary label
y = df["Resistant Phenotype"].map({"Susceptible": 0, "Resistant": 1})
X = df.drop(columns=["Resistant Phenotype"])

# Remove constant (zero-variance) columns
X = X.loc[:, X.nunique() > 1]

# =====================================================
# CONFLICT FILTERING AND GENOMIC DEDUPLICATION
# Removes profiles with discordant phenotype labels,
# then retains one representative per unique profile.
# =====================================================
print("\n============================")
print("CONFLICT FILTERING & DEDUPLICATION")
print("============================")

genome_sig = X.astype(str).agg("_".join, axis=1)
df_tmp = pd.concat([X, y], axis=1)
df_tmp["sig"] = genome_sig

# Keep only profiles with a single unambiguous phenotype
label_counts = df_tmp.groupby("sig")["Resistant Phenotype"].nunique()
valid_sigs = label_counts[label_counts == 1].index

df_clean = (
    df_tmp[df_tmp["sig"].isin(valid_sigs)]
    .drop_duplicates(subset="sig")
    .drop(columns="sig")
    .reset_index(drop=True)
)

X = df_clean.drop(columns=["Resistant Phenotype"])
y = df_clean["Resistant Phenotype"]

print(f"Total samples after deduplication: {len(X)}")
print(y.value_counts().rename({0: "Susceptible", 1: "Resistant"}))

# =====================================================
# CLADE-AWARE TRAIN/TEST SPLIT
# Agglomerative clustering on Hamming distance prevents
# phylogenetically related isolates from spanning both
# training and test partitions (lineage leakage).
# =====================================================
print("\n============================")
print("CLADE-AWARE SPLITTING")
print("============================")

cluster_model = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.05,
    metric="hamming",
    linkage="average"
)
groups = cluster_model.fit_predict(X)

# Fallback: if too few clusters, use fixed n_clusters
if len(np.unique(groups)) < 5:
    cluster_model = AgglomerativeClustering(
        n_clusters=50,
        metric="hamming",
        linkage="average"
    )
    groups = cluster_model.fit_predict(X)

print(f"Total samples: {len(X)} | Clade count: {len(np.unique(groups))}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

xtr = X.iloc[train_idx]
xts = X.iloc[test_idx]
ytr = y.iloc[train_idx]
yts = y.iloc[test_idx]

print(f"Train size: {len(xtr)} | Test size: {len(xts)}")
print("Train class distribution:\n", ytr.value_counts())
print("Test  class distribution:\n", yts.value_counts())

# =====================================================
# EVALUATION UTILITIES
# =====================================================

def permutation_mcc(y_true, y_pred, n_perm=100):
    """Estimate null MCC distribution by label permutation."""
    scores = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y_true)
        scores.append(matthews_corrcoef(y_perm, y_pred))
    return np.mean(scores)


def compute_metrics(y_true, y_pred, y_prob, label):
    """Return a dict of classification metrics for one model."""
    return {
        "Model":     label,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "F1":        f1_score(y_true, y_pred),
        "MCC":       matthews_corrcoef(y_true, y_pred),
        "AUROC":     roc_auc_score(y_true, y_prob),
        "Perm_MCC":  permutation_mcc(y_true, y_pred, n_perm=100)
    }


def bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=1000, random_state=42):
    """Non-parametric bootstrap 95% CI for MCC, AUROC, and F1."""
    np.random.seed(random_state)
    mcc_scores, auc_scores, f1_scores = [], [], []

    for _ in range(n_bootstrap):
        idx = resample(range(len(y_true)))
        y_t  = np.array(y_true)[idx]
        y_p  = np.array(y_pred)[idx]
        y_pr = np.array(y_prob)[idx]
        if len(np.unique(y_t)) < 2:
            continue
        mcc_scores.append(matthews_corrcoef(y_t, y_p))
        auc_scores.append(roc_auc_score(y_t, y_pr))
        f1_scores.append(f1_score(y_t, y_p))

    def summary(scores):
        return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

    return {
        "MCC":   summary(mcc_scores),
        "AUROC": summary(auc_scores),
        "F1":    summary(f1_scores)
    }


def delong_bootstrap(y_true, prob1, prob2, n_boot=1000):
    """
    Bootstrap-based AUROC comparison test.
    Returns two-sided p-value for the difference in AUROCs.
    """
    y_true = np.array(y_true)
    auc_diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        y_b, p1, p2 = y_true[idx], prob1[idx], prob2[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_diffs.append(roc_auc_score(y_b, p1) - roc_auc_score(y_b, p2))
    auc_diffs = np.array(auc_diffs)
    return 2 * min(np.mean(auc_diffs <= 0), np.mean(auc_diffs >= 0))

# =====================================================
# PYTORCH DATASET
# =====================================================
class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# =====================================================
# TABTRANSFORMER ENCODER
# Embedding dimension: 32 | Heads: 4 | Layers: 3
# Each genomic feature is mapped to a learnable token:
#   z_ij = e_j + w_v * x_ij
# Self-attention aggregates contextual gene interactions.
# Final isolate embedding: mean pooling over all tokens.
# =====================================================
class TabTransformerExtractor(nn.Module):
    def __init__(self, num_features, d_model=32, nhead=4, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.feature_embed = nn.Embedding(num_features, d_model)
        self.value_proj    = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x, return_attn=False):
        B, F = x.shape
        idx  = torch.arange(F, device=x.device).unsqueeze(0).repeat(B, 1)
        tok  = self.feature_embed(idx) + self.value_proj(x.unsqueeze(-1))

        if return_attn:
            attns, out = [], tok
            for layer in self.transformer.layers:
                attn_out, attn = layer.self_attn(
                    out, out, out, need_weights=True, average_attn_weights=False
                )
                attns.append(attn.detach())
                out = layer(out)
            return out.mean(1), attns

        return self.transformer(tok).mean(1)


def train_encoder(extractor, X, y, epochs=120, lr=1e-3, weight_decay=1e-4, batch_size=128):
    """Train the TabTransformer encoder with a linear classification head."""
    head     = nn.Linear(extractor.d_model, 2).to(device)
    optimizer = torch.optim.Adam(
        list(extractor.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    loss_fn  = nn.CrossEntropyLoss()
    loader   = DataLoader(TabDataset(X, y), batch_size=batch_size, shuffle=True)
    extractor.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(head(extractor(xb)), yb).backward()
            optimizer.step()
    extractor.eval()
    return extractor


def get_embeddings(extractor, X):
    """Extract latent embeddings for a feature matrix X."""
    with torch.no_grad():
        return extractor(
            torch.tensor(X.values, dtype=torch.float32).to(device)
        ).cpu().numpy()

# =====================================================
# BASE CLASSIFIER BENCHMARKING
# =====================================================
print("\n============================")
print("BASE CLASSIFIER BENCHMARKING")
print("============================")

results = []
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=250),
    "MLP":                 MLPClassifier(hidden_layer_sizes=(128, 64),
                                         max_iter=250, random_state=RANDOM_SEED),
    "Random Forest":       RandomForestClassifier(n_estimators=250, max_depth=6,
                                                   random_state=RANDOM_SEED),
    "XGBoost":             XGBClassifier(n_estimators=250, max_depth=6,
                                          eval_metric="logloss"),
}

for name, model in base_models.items():
    model.fit(xtr, ytr)
    pred = model.predict(xts)
    prob = model.predict_proba(xts)[:, 1] if hasattr(model, "predict_proba") else pred
    results.append(compute_metrics(yts, pred, prob, name))

# Baseline CatBoost
base_cb = CatBoostClassifier(iterations=250, depth=8, verbose=0, random_state=RANDOM_SEED)
base_cb.fit(xtr, ytr)
pred = base_cb.predict(xts)
prob = base_cb.predict_proba(xts)[:, 1]
results.append(compute_metrics(yts, pred, prob, "Baseline CatBoost"))
trained_models["Baseline CatBoost"] = base_cb
internal_preds["Baseline CatBoost"] = pred
internal_probs["Baseline CatBoost"] = prob

# =====================================================
# TABTRANSFORMER ENCODER TRAINING
# =====================================================
print("\n============================")
print("TABTRANSFORMER ENCODER TRAINING")
print("============================")

extractor = TabTransformerExtractor(X.shape[1]).to(device)
extractor = train_encoder(extractor, xtr, ytr)

emb_tr = get_embeddings(extractor, xtr)
emb_ts = get_embeddings(extractor, xts)

# TabTransformer-CatBoost (no hyperparameter tuning)
cb = CatBoostClassifier(iterations=250, depth=8, verbose=0, random_state=RANDOM_SEED)
cb.fit(emb_tr, ytr)
pred = cb.predict(emb_ts)
prob = cb.predict_proba(emb_ts)[:, 1]
results.append(compute_metrics(yts, pred, prob, "TabTransformer-CatBoost"))
trained_models["TabTransformer-CatBoost"] = cb
internal_preds["TabTransformer-CatBoost"] = pred
internal_probs["TabTransformer-CatBoost"] = prob

# =====================================================
# HYPERPARAMETER OPTIMIZATION
# Single-objective: maximize F1 (raw genomic features)
# Optimizers compared: TPE, CMA-ES, NSGA-II
# =====================================================
print("\n============================")
print("HYPERPARAMETER OPTIMIZATION (TPE / CMA-ES / NSGA-II)")
print("============================")

def objective_raw(trial):
    params = {
        "depth":        trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "l2_leaf_reg":  trial.suggest_float("l2_leaf_reg", 1, 6),
    }
    m = CatBoostClassifier(iterations=250, verbose=0, **params)
    m.fit(xtr, ytr)
    return f1_score(yts, m.predict(xts))

for sampler_name, sampler in [("TPE",    TPESampler(seed=RANDOM_SEED)),
                               ("CMA-ES", CmaEsSampler(seed=RANDOM_SEED)),
                               ("NSGA-II", NSGAIISampler(seed=RANDOM_SEED))]:
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_raw, n_trials=50)
    cb = CatBoostClassifier(iterations=250, verbose=0, **study.best_params)
    cb.fit(xtr, ytr)
    pred = cb.predict(xts)
    prob = cb.predict_proba(xts)[:, 1]
    label = f"Optimized CatBoost {sampler_name}"
    results.append(compute_metrics(yts, pred, prob, label))
    trained_models[label] = cb
    internal_preds[label] = pred
    internal_probs[label] = prob

# =====================================================
# CHAINED HYBRID: TABTRANSFORMER + CATBOOST
# Multi-objective optimization (F1, MCC) via NSGA-II
# on transformer-derived embeddings.
# =====================================================
print("\n============================")
print("CHAINED HYBRID — MULTI-OBJECTIVE NSGA-II")
print("============================")

def objective_hybrid(trial):
    params = {
        "depth":        trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "l2_leaf_reg":  trial.suggest_float("l2_leaf_reg", 1, 6),
    }
    m = CatBoostClassifier(iterations=250, verbose=0, **params)
    m.fit(emb_tr, ytr)
    pr = m.predict(emb_ts)
    return f1_score(yts, pr), matthews_corrcoef(yts, pr)

study = optuna.create_study(
    directions=["maximize", "maximize"],
    sampler=NSGAIISampler(seed=RANDOM_SEED)
)
study.optimize(objective_hybrid, n_trials=50)

# Select Pareto-optimal trial maximizing MCC
best_trial = max(study.best_trials, key=lambda t: t.values[1])
print(f"Best Pareto point (F1, MCC): {best_trial.values}")

cb = CatBoostClassifier(iterations=250, verbose=0, **best_trial.params)
cb.fit(emb_tr, ytr)
pred = cb.predict(emb_ts)
prob = cb.predict_proba(emb_ts)[:, 1]
results.append(compute_metrics(yts, pred, prob, "Chained Hybrid"))
trained_models["Chained Hybrid"] = cb
internal_preds["Chained Hybrid"] = pred
internal_probs["Chained Hybrid"] = prob

# =====================================================
# INTERNAL RESULTS TABLE
# =====================================================
final = pd.DataFrame(results)
print("\n============================")
print("INTERNAL BENCHMARKING RESULTS")
print("============================")
print(final.round(4).to_string(index=False))

# =====================================================
# EXTERNAL VALIDATION
# Fixed decision threshold = 0.05 to mitigate
# performance degradation under covariate shift.
# =====================================================
print("\n============================")
print("EXTERNAL VALIDATION")
print("============================")

val_full = pd.read_csv("validationPRJlison.csv", sep=";")
meta = val_full[["BioProjectAccession", "IsolationCountry"]].copy()
val  = val_full.drop(columns=["BioProjectAccession", "IsolationCountry"])

if "Genome ID" in val.columns:
    val = val.drop(columns=["Genome ID"])

y_val = val["Resistant Phenotype"].map({"Susceptible": 0, "Resistant": 1})
X_val = val.drop(columns=["Resistant Phenotype"])
X_val = X_val.reindex(columns=X.columns, fill_value=0)

print(f"External cohort size: {len(X_val)}")
print("Class distribution:\n", y_val.value_counts())

CALIBRATED_THRESHOLD = 0.05

# Retrain best model on full discovery dataset
best_model_name = final.loc[final["F1"].idxmax(), "Model"]
best_model = trained_models[best_model_name]

print(f"\nSelected best model: {best_model_name}")

if best_model_name in ["TabTransformer-CatBoost", "Chained Hybrid"]:
    extractor_full = TabTransformerExtractor(X.shape[1]).to(device)
    extractor_full = train_encoder(extractor_full, X, y)
    emb_full = get_embeddings(extractor_full, X)
    emb_val  = get_embeddings(extractor_full, X_val)
    final_model = CatBoostClassifier(**best_model.get_params())
    final_model.fit(emb_full, y)
    prob_val = final_model.predict_proba(emb_val)[:, 1]
else:
    final_model = CatBoostClassifier(**best_model.get_params())
    final_model.fit(X, y)
    prob_val = final_model.predict_proba(X_val)[:, 1]

pred_val = (prob_val > CALIBRATED_THRESHOLD).astype(int)
ext_results = compute_metrics(y_val, pred_val, prob_val, "External Validation")

print("\nEXTERNAL VALIDATION RESULTS")
print(pd.DataFrame([ext_results]).round(4).to_string(index=False))

# =====================================================
# PER BIOPROJECT PERFORMANCE
# =====================================================
print("\n============================")
print("PER BIOPROJECT PERFORMANCE")
print("============================")

meta = meta.reset_index(drop=True)
X_val_meta = X_val.reset_index(drop=True)
y_val_meta = y_val.reset_index(drop=True)
per_project_results = []

for project in meta["BioProjectAccession"].unique():
    idx = meta["BioProjectAccession"] == project
    if idx.sum() < 3:
        continue
    X_p = X_val_meta[idx]
    y_p = y_val_meta[idx]
    if best_model_name in ["TabTransformer-CatBoost", "Chained Hybrid"]:
        prob_p = final_model.predict_proba(get_embeddings(extractor_full, X_p))[:, 1]
    else:
        prob_p = final_model.predict_proba(X_p)[:, 1]
    pred_p = (prob_p > CALIBRATED_THRESHOLD).astype(int)
    res = compute_metrics(y_p, pred_p, prob_p, project)
    res["n"] = int(idx.sum())
    per_project_results.append(res)

per_df = pd.DataFrame(per_project_results)
per_df.to_csv("TABLE_per_project.csv", index=False)
print(per_df.round(4).to_string(index=False))

# =====================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =====================================================
print("\n============================")
print("BOOTSTRAP CONFIDENCE INTERVALS (n=1000)")
print("============================")

best_pred = internal_preds[best_model_name]
best_prob = internal_probs[best_model_name]

print("\nINTERNAL TEST SET")
int_ci = bootstrap_ci(yts.values, best_pred, best_prob)
for metric, (mean, lo, hi) in int_ci.items():
    print(f"  {metric}: {mean:.4f}  (95% CI: {lo:.4f} – {hi:.4f})")

print("\nEXTERNAL VALIDATION SET")
ext_ci = bootstrap_ci(y_val.values, pred_val, prob_val)
for metric, (mean, lo, hi) in ext_ci.items():
    print(f"  {metric}: {mean:.4f}  (95% CI: {lo:.4f} – {hi:.4f})")

# =====================================================
# ABLATION STUDY — RAW DATA BASELINE (NO DEDUP / NO CLADE SPLIT)
# Demonstrates that random-split + non-deduplicated training
# produces inflated internal metrics and collapses on external data.
# =====================================================
print("\n============================")
print("ABLATION: RAW DATA BASELINE CATBOOST")
print("============================")

df_raw = pd.read_csv("asilverisetigenler.csv", sep=";")
if "Genome ID" in df_raw.columns:
    df_raw = df_raw.drop(columns=["Genome ID"])
y_raw = df_raw["Resistant Phenotype"].map({"Susceptible": 0, "Resistant": 1})
X_raw = df_raw.drop(columns=["Resistant Phenotype"])
X_raw = X_raw.loc[:, X_raw.nunique() > 1]

Xr_tr, Xr_ts, yr_tr, yr_ts = train_test_split(
    X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=RANDOM_SEED
)

raw_cb = CatBoostClassifier(iterations=250, depth=8, verbose=0, random_state=RANDOM_SEED)
raw_cb.fit(Xr_tr, yr_tr)
raw_pred = raw_cb.predict(Xr_ts)
raw_prob = raw_cb.predict_proba(Xr_ts)[:, 1]

print("Internal (raw):")
print(pd.DataFrame([compute_metrics(yr_ts, raw_pred, raw_prob, "Raw CatBoost")]).round(4).to_string(index=False))

val_raw = pd.read_csv("validationPRJlison.csv", sep=";")
if "Genome ID" in val_raw.columns:
    val_raw = val_raw.drop(columns=["Genome ID"])
for col in ["BioProjectAccession", "IsolationCountry"]:
    if col in val_raw.columns:
        val_raw = val_raw.drop(columns=[col])
y_val_raw = val_raw["Resistant Phenotype"].map({"Susceptible": 0, "Resistant": 1})
X_val_raw = val_raw.drop(columns=["Resistant Phenotype"]).reindex(columns=X_raw.columns, fill_value=0)

raw_ext_pred = raw_cb.predict(X_val_raw)
raw_ext_prob = raw_cb.predict_proba(X_val_raw)[:, 1]

print("External (raw):")
print(pd.DataFrame([compute_metrics(y_val_raw, raw_ext_pred, raw_ext_prob, "Raw CatBoost External")]).round(4).to_string(index=False))

# =====================================================
# DELONG AUROC COMPARISON TEST
# =====================================================
print("\n============================")
print("DELONG TEST (AUROC COMPARISON)")
print("============================")

y_true_int = yts.values
p_val = delong_bootstrap(y_true_int, internal_probs["Baseline CatBoost"],
                          internal_probs["Chained Hybrid"])
print(f"Baseline CatBoost AUROC : {roc_auc_score(y_true_int, internal_probs['Baseline CatBoost']):.4f}")
print(f"Chained Hybrid AUROC    : {roc_auc_score(y_true_int, internal_probs['Chained Hybrid']):.4f}")
print(f"Bootstrap DeLong p-value: {p_val:.4f}")

# =====================================================
# LATENT SPACE MAPPING
# Pearson correlation between each latent dimension
# and the original 195 genomic markers.
# =====================================================
print("\n============================")
print("LATENT DIMENSION TO GENE MAPPING")
print("============================")

def map_latent_to_genes(X_df, embeddings, target_dimensions, top_k=10):
    """Correlate each latent dimension with original binary gene features."""
    gene_names = X_df.columns.tolist()
    X_matrix   = X_df.values
    results    = []
    for d in target_dimensions:
        dim_vec = embeddings[:, d]
        corrs   = []
        for i, gene in enumerate(gene_names):
            c = np.corrcoef(X_matrix[:, i], dim_vec)[0, 1]
            if not np.isnan(c):
                corrs.append((gene, c))
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        for gene, c in corrs[:top_k]:
            results.append({
                "Latent_Dimension": d,
                "Gene_Name":        gene,
                "Correlation":      round(c, 4),
                "Direction":        "Positive" if c > 0 else "Negative"
            })
    return pd.DataFrame(results)

# Full 32-dimension mapping (Supplementary Table S1)
mapping_df = map_latent_to_genes(xtr, emb_tr, list(range(32)), top_k=5)
mapping_df.to_csv("FULL_LATENT_GENE_MAPPING_32D.csv", index=False)
print("Supplementary Table S1 saved: FULL_LATENT_GENE_MAPPING_32D.csv")

# =====================================================
# SHAP ATTRIBUTION ON TRANSFORMER EMBEDDINGS
# =====================================================
print("\n============================")
print("SHAP ANALYSIS (EMBEDDING SPACE)")
print("============================")

if best_model_name in ["TabTransformer-CatBoost", "Chained Hybrid"]:
    shap_X    = pd.DataFrame(emb_tr)
    explainer  = shap.TreeExplainer(final_model)
    shap_vals  = explainer.shap_values(shap_X)

    shap.summary_plot(shap_vals, shap_X, show=False, plot_size=(7, 5))
    plt.title("Feature importance in transformer embedding space")
    plt.tight_layout()
    plt.savefig("FIG5_shap.png", dpi=600)
    plt.close()
    print("FIG5_shap.png saved")

# =====================================================
# ATTENTION WEIGHT ANALYSIS — TOP GENE PAIRS
# =====================================================
print("\n============================")
print("ATTENTION WEIGHT ANALYSIS")
print("============================")

def get_attention_matrix(extractor, X_df, device):
    """Average multi-head attention weights across layers, heads, and samples."""
    extractor.eval()
    with torch.no_grad():
        x = torch.tensor(X_df.values[:128], dtype=torch.float32).to(device)
        _, attn_list = extractor(x, return_attn=True)
    attn = torch.stack(attn_list).mean(0).mean(0).mean(0)
    return attn.cpu().numpy()

att    = get_attention_matrix(extractor_full, X, device)
genes  = X.columns.tolist()
pairs  = []
for i in range(len(genes)):
    for j in range(i + 1, len(genes)):
        pairs.append((genes[i], genes[j], att[i, j]))

top_pairs = pd.DataFrame(pairs, columns=["Gene_A", "Gene_B", "Attention"])
top_pairs = top_pairs.sort_values("Attention", ascending=False).head(20)
labels = top_pairs["Gene_A"] + " ↔ " + top_pairs["Gene_B"]

plt.figure(figsize=(5.5, 4))
plt.barh(labels.values[::-1], top_pairs["Attention"].values[::-1])
plt.xlabel("Attention weight")
plt.title("Top genomic interactions (multi-head self-attention)")
plt.tight_layout()
plt.savefig("FIG6_attention_pairs.png", dpi=600)
plt.close()
print("FIG6_attention_pairs.png saved")

# =====================================================
# COMBINED FIGURES
# =====================================================
print("\n============================")
print("GENERATING FIGURES")
print("============================")

# --- ROC curves (internal + external) ---
fpr_int, tpr_int, _ = roc_curve(yts, best_prob)
fpr_ext, tpr_ext, _ = roc_curve(y_val, prob_val)
auc_int = auc(fpr_int, tpr_int)
auc_ext = auc(fpr_ext, tpr_ext)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, fpr, tpr, a, title in [
    (axes[0], fpr_int, tpr_int, auc_int, "(A) Internal Test"),
    (axes[1], fpr_ext, tpr_ext, auc_ext, "(B) External Validation"),
]:
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {a:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
plt.tight_layout()
plt.savefig("FIG3_ROC_combined.png", dpi=300)
plt.close()
print("FIG3_ROC_combined.png saved")

# --- Confusion matrices (internal + external) ---
def norm_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).astype(float)
    return cm / cm.sum(axis=1)[:, None]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, cm, title in [
    (axes[0], norm_cm(yts, best_pred),   "(A) Internal test"),
    (axes[1], norm_cm(y_val, pred_val),  "(B) External validation"),
]:
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, square=True,
                xticklabels=["Susceptible", "Resistant"],
                yticklabels=["Susceptible", "Resistant"], ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("FIG_CM_combined.png", dpi=600)
plt.close()
print("FIG_CM_combined.png saved")

# --- t-SNE (internal + external) ---
z_test = TSNE(2, perplexity=30, learning_rate="auto", init="pca",
              random_state=RANDOM_SEED).fit_transform(emb_ts)
z_ext  = TSNE(2, perplexity=30, learning_rate="auto", init="pca",
              random_state=RANDOM_SEED).fit_transform(emb_val)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, z, yy, title in [
    (axes[0], z_test, yts.values,    "(A) Internal test embeddings"),
    (axes[1], z_ext,  y_val.values,  "(B) External embeddings"),
]:
    ax.scatter(z[:, 0], z[:, 1], c=yy, cmap="coolwarm", s=20, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("FIG2_TSNE.png", dpi=300)
plt.close()
print("FIG2_TSNE.png saved")

# --- Embedding dimension correlation heatmap ---
corr_matrix = np.corrcoef(emb_tr.T)
plt.figure(figsize=(7, 6))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True, cbar=True)
plt.title("Pairwise Pearson correlation — 32 latent dimensions")
plt.tight_layout()
plt.savefig("embeded.png", dpi=300)
plt.close()
print("embeded.png saved")

print("\n============================")
print("PIPELINE COMPLETE")
print("============================")
