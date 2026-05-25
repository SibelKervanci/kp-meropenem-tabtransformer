# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:59:10 2026

@author: Sibel Kervancı

Revision: Added ablation comparisons per Reviewer 1, Comment 1:
  - Embedding + Logistic Regression
  - Embedding + MLP  
  - Transformer-Only (end-to-end with built-in classification head)
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,confusion_matrix, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from catboost import CatBoostClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupShuffleSplit
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
import shap
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
trained_models = {}
internal_preds = {}
internal_probs = {}
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 11        # genel
matplotlib.rcParams['axes.titlesize'] = 12   # başlık
matplotlib.rcParams['axes.labelsize'] = 11   # eksen etiketi
matplotlib.rcParams['xtick.labelsize'] = 10  # x tick
matplotlib.rcParams['ytick.labelsize'] = 10  # y tick
# =====================================================
# DATA
# =====================================================
df = pd.read_csv("kp_meropenem_training.csv", sep=";")

if "Genome ID" in df.columns:
    df = df.drop(columns=["Genome ID"])

# label encode
y = df["Resistant Phenotype"].map({"Susceptible":0,"Resistant":1})
X = df.drop(columns=["Resistant Phenotype"])

# sabit sütunları sil
X = X.loc[:, X.nunique() > 1]

# =========================================
# SAFE DEDUP (ÇELİŞKİLİ GENOMLARI TAMAMEN SİL)
# =========================================

# genom imzası
genome_sig = X.astype(str).agg("_".join, axis=1)

df_tmp = pd.concat([X, y], axis=1)
df_tmp["sig"] = genome_sig

# her genom için kaç farklı phenotype var?
label_counts = df_tmp.groupby("sig")["Resistant Phenotype"].nunique()

# SADECE tek phenotype olan genomları tut
valid_sigs = label_counts[label_counts == 1].index

df_clean = (
    df_tmp[df_tmp["sig"].isin(valid_sigs)]
    .drop_duplicates(subset="sig")
    .drop(columns="sig")
    .reset_index(drop=True)
)

# final X y
X = df_clean.drop(columns=["Resistant Phenotype"])
y = df_clean["Resistant Phenotype"]

# X, y safe-dedup sonrası hazır olmalı
# X: sadece gen sütunları (binary matrix)
# y: 0/1 phenotype

# --- CLUSTER OLUŞTUR ---
# Hamming mesafesi: binary gen profilleri için doğru seçim
cluster_model = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.05,   # benzerlik eşiği (tune edilebilir)
    metric="hamming",
    linkage="average"
)

groups = cluster_model.fit_predict(X)

# Çok az grup çıkarsa fallback
if len(np.unique(groups)) < 5:
    cluster_model = AgglomerativeClustering(
        n_clusters=50,
        metric="hamming",
        linkage="average"
    )
    groups = cluster_model.fit_predict(X)

print(f"Total samples: {len(X)} | Clade count: {len(np.unique(groups))}")


# split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, test_idx = next(gss.split(X, y, groups=groups))

xtr = X.iloc[train_idx]
xts = X.iloc[test_idx]
ytr = y.iloc[train_idx]
yts = y.iloc[test_idx]

print("Train size:", len(xtr))
print("Test size:", len(xts))
print("Train class dist:\n", ytr.value_counts())
print("Test class dist:\n", yts.value_counts())

print("Total samples after SAFE dedup:", len(X))
print(y.value_counts().rename({0:"Susceptible",1:"Resistant"}))


# =====================================================
# PERM MCC
# =====================================================
def perm_mcc(y_true, y_pred, n_perm=100):
    scores = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y_true)
        scores.append(matthews_corrcoef(y_perm, y_pred))
    return np.mean(scores)

def metrics(y_true,y_pred,y_prob,label):
    return {
        "Model":label,
        "Accuracy":accuracy_score(y_true,y_pred),
        "Precision":precision_score(y_true,y_pred),
        "Recall":recall_score(y_true,y_pred),
        "F1":f1_score(y_true,y_pred),
        "MCC":matthews_corrcoef(y_true,y_pred),
        "AUROC":roc_auc_score(y_true,y_prob),
        "Perm_MCC": perm_mcc(y_true, y_pred, n_perm=100)
    }
# =====================================================
# PLOTTING
# =====================================================

def plot_roc(y_true, y_prob, name):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"roc_{name}.png", dpi=300)
    plt.close()
# =====================================================
# DATASET
# =====================================================
class TabDataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X.values,dtype=torch.float32)
        self.y=torch.tensor(y.values,dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.y[i]

# =====================================================
# TRANSFORMER
# =====================================================
class TabTransformerExtractor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.d_model = 32

        self.feature_embed = nn.Embedding(num_features, self.d_model)
        self.value_proj = nn.Linear(1, self.d_model)

        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=self.d_model*2,
            dropout=0.2,
            batch_first=True
        )

        self.tr = nn.TransformerEncoder(self.enc_layer, num_layers=3)

    def forward(self, x, return_attn=False):

        B, F = x.shape
        idx = torch.arange(F, device=x.device).unsqueeze(0).repeat(B,1)

        feat = self.feature_embed(idx)
        val  = self.value_proj(x.unsqueeze(-1))
        tok  = feat + val

        if return_attn:
            attns = []
            out = tok

            for layer in self.tr.layers:
                attn_out, attn = layer.self_attn(
                   out, out, out,
                   need_weights=True,
                   average_attn_weights=False
                )
                attns.append(attn.detach())
                out = layer(out)

            return out.mean(1), attns

        out = self.tr(tok)
        return out.mean(1)

def train_extractor(extractor,X,y,epochs=120):

    head=nn.Linear(extractor.d_model,2).to(device)

    opt=torch.optim.Adam(
        list(extractor.parameters())+list(head.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    loss_fn=nn.CrossEntropyLoss()

    loader=DataLoader(
        TabDataset(X,y),
        batch_size=128,
        shuffle=True
    )

    extractor.train()

    for _ in range(epochs):
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)

            opt.zero_grad()
            out=head(extractor(xb))
            loss=loss_fn(out,yb)
            loss.backward()
            opt.step()

    extractor.eval()
    return extractor

def get_emb(ext,X):
    with torch.no_grad():
        return ext(torch.tensor(X.values,dtype=torch.float32).to(device)).cpu().numpy()

# =====================================================
# RESULTS
# =====================================================
results=[]
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=250),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=250, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=6, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=250, max_depth=6, eval_metric="logloss"),
}

for name, model in base_models.items():
    model.fit(xtr, ytr)
    pred = model.predict(xts)

    if hasattr(model,"predict_proba"):
        prob = model.predict_proba(xts)[:,1]
    else:
        prob = pred

    results.append(metrics(yts,pred,prob,name))
# BASE CATBOOST
base=CatBoostClassifier(iterations=250,depth=8,verbose=0,random_state=42)
base.fit(xtr,ytr)
trained_models["Baseline CatBoost"] = base
pred=base.predict(xts)
prob=base.predict_proba(xts)[:,1]
results.append(metrics(yts,pred,prob,"Baseline CatBoost"))
internal_preds["Baseline CatBoost"] = pred
internal_probs["Baseline CatBoost"] = prob
# =====================================================
# TRANSFORMER + CATBOOST
# =====================================================
extractor = TabTransformerExtractor(X.shape[1]).to(device)
extractor = train_extractor(extractor,xtr,ytr)

emb_tr = get_emb(extractor,xtr)
emb_ts = get_emb(extractor,xts)

cb = CatBoostClassifier(iterations=250, depth=8,verbose=0, random_state=42)

cb.fit(emb_tr,ytr)
pred=cb.predict(emb_ts)
prob=cb.predict_proba(emb_ts)[:,1]

results.append(metrics(yts,pred,prob,"TabTransformer-CatBoost"))
trained_models["TabTransformer-CatBoost"] = cb
internal_preds["TabTransformer-CatBoost"] = pred
internal_probs["TabTransformer-CatBoost"] = prob

# =====================================================
# REVIEWER 1 – COMMENT 1: ABLATION COMPARISONS
# Embedding + Logistic Regression
# Embedding + MLP
# Transformer-Only (with built-in classification head)
# =====================================================

print("\n==============================")
print("ABLATION: EMBEDDING-BASED CLASSIFIERS")
print("==============================")

# --- 1. Embedding + Logistic Regression ---
emb_lr = LogisticRegression(max_iter=500, random_state=42)
emb_lr.fit(emb_tr, ytr)
pred_emb_lr = emb_lr.predict(emb_ts)
prob_emb_lr = emb_lr.predict_proba(emb_ts)[:, 1]
results.append(metrics(yts, pred_emb_lr, prob_emb_lr, "Embedding + LR"))
trained_models["Embedding + LR"] = emb_lr
internal_preds["Embedding + LR"] = pred_emb_lr
internal_probs["Embedding + LR"] = prob_emb_lr

# --- 2. Embedding + MLP ---
emb_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
emb_mlp.fit(emb_tr, ytr)
pred_emb_mlp = emb_mlp.predict(emb_ts)
prob_emb_mlp = emb_mlp.predict_proba(emb_ts)[:, 1]
results.append(metrics(yts, pred_emb_mlp, prob_emb_mlp, "Embedding + MLP"))
trained_models["Embedding + MLP"] = emb_mlp
internal_preds["Embedding + MLP"] = pred_emb_mlp
internal_probs["Embedding + MLP"] = prob_emb_mlp

# --- 3. Transformer-Only (end-to-end, kendi classification head'i ile) ---
class TabTransformerClassifier(nn.Module):
    """
    Standalone TabTransformer with a built-in classification head.
    Used as a Transformer-Only baseline (no external CatBoost classifier).
    The encoder is identical to TabTransformerExtractor to ensure
    a fair architectural comparison.
    """
    def __init__(self, num_features, d_model=32, nhead=4, num_layers=3,
                 dropout=0.2, num_classes=2):
        super().__init__()
        self.d_model = d_model
        # Feature identity embedding (same as extractor)
        self.feature_embed = nn.Embedding(num_features, d_model)
        # Scalar value projection (same as extractor)
        self.value_proj = nn.Linear(1, d_model)
        # Transformer encoder (same config as extractor)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        # Classification head (replaces CatBoost)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        B, F = x.shape
        idx = torch.arange(F, device=x.device).unsqueeze(0).repeat(B, 1)
        feat = self.feature_embed(idx)                  # (B, F, d_model)
        val  = self.value_proj(x.unsqueeze(-1))         # (B, F, d_model)
        tok  = feat + val                               # (B, F, d_model)
        out  = self.tr(tok)                             # (B, F, d_model)
        pooled = out.mean(1)                            # (B, d_model) — mean pooling
        return self.head(pooled)                        # (B, num_classes)


def train_transformer_only(X_train, y_train, num_features, epochs=120):
    """Train the standalone TabTransformerClassifier end-to-end."""
    model = TabTransformerClassifier(num_features).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TabDataset(X_train, y_train), batch_size=128, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_transformer_only(model, X_df):
    """Return (predictions, probabilities) for a given DataFrame."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_df.values, dtype=torch.float32).to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = (probs > 0.5).astype(int)
    return preds, probs


# Train Transformer-Only model
tr_only_model = train_transformer_only(xtr, ytr, X.shape[1])

# Predict on internal test set
pred_tr_only, prob_tr_only = predict_transformer_only(tr_only_model, xts)

results.append(metrics(yts, pred_tr_only, prob_tr_only, "Transformer-Only"))
internal_preds["Transformer-Only"] = pred_tr_only
internal_probs["Transformer-Only"] = prob_tr_only

print("Ablation models trained successfully.")
print("  - Embedding + LR")
print("  - Embedding + MLP")
print("  - Transformer-Only")

# =====================================================
# OBJECTIVE CATBOOST
# =====================================================
def obj(trial):
    p={
        "depth":trial.suggest_int("depth",4,8),
        "learning_rate":trial.suggest_float("learning_rate",0.01,0.1),
        "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1,6)
    }
    m=CatBoostClassifier(iterations=250,verbose=0,**p)
    m.fit(xtr,ytr)
    pr=m.predict(xts)
    return f1_score(yts,pr)

# TPE
study=optuna.create_study(direction="maximize",sampler=TPESampler(seed=42))
study.optimize(obj,n_trials=50)
cb=CatBoostClassifier(iterations=250,verbose=0,**study.best_params)
cb.fit(xtr,ytr)
pred=cb.predict(xts)
prob=cb.predict_proba(xts)[:,1]
results.append(metrics(yts,pred,prob,"Optimized CatBoost TPE"))
trained_models["Optimized CatBoost TPE"] = cb
internal_preds["Optimized CatBoost TPE"] = pred
internal_probs["Optimized CatBoost TPE"] = prob
# CMA
study=optuna.create_study(direction="maximize",sampler=CmaEsSampler(seed=42))
study.optimize(obj,n_trials=50)
cb=CatBoostClassifier(iterations=250,verbose=0,**study.best_params)
cb.fit(xtr,ytr)
pred=cb.predict(xts)
prob=cb.predict_proba(xts)[:,1]
results.append(metrics(yts,pred,prob,"Optimized CatBoost CMA-ES"))
trained_models["Optimized CatBoost CMA-ES"] = cb
internal_preds["Optimized CatBoost CMA-ES"] = pred
internal_probs["Optimized CatBoost CMA-ES"] = prob
# NSGA
study=optuna.create_study(direction="maximize",sampler=NSGAIISampler(seed=42))
study.optimize(obj,n_trials=50)
cb=CatBoostClassifier(iterations=250,verbose=0,**study.best_params)
cb.fit(xtr,ytr)
pred=cb.predict(xts)
prob=cb.predict_proba(xts)[:,1]
results.append(metrics(yts,pred,prob,"Optimized CatBoost NSGA II"))
trained_models["Optimized CatBoost NSGA II"] = cb
internal_preds["Optimized CatBoost NSGA II"] = pred
internal_probs["Optimized CatBoost NSGA II"] = prob

# =====================================================
# CHAINED HYBRID (MULTI-OBJECTIVE NSGA-II OPTIMIZATION)
# =====================================================
def obj_emb(trial):
    p = {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 6)
    }
    
    m = CatBoostClassifier(iterations=250, verbose=0, **p)
    m.fit(emb_tr, ytr)
    
    pr = m.predict(emb_ts)
    
    # Çok amaçlı optimizasyon için iki metrik döndürüyoruz
    f1 = f1_score(yts, pr)
    mcc = matthews_corrcoef(yts, pr)
    
    return f1, mcc

# Optuna'da iki yönlü maksimizasyon tanımlıyoruz
study = optuna.create_study(
    directions=["maximize", "maximize"], 
    sampler=NSGAIISampler(seed=42)   
)
study.optimize(obj_emb, n_trials=50)

# Pareto cephesindeki (Pareto Front) en iyi denge noktalarından birini seçiyoruz
# Burada genellikle en yüksek MCC veren trial seçilir
best_trial = max(study.best_trials, key=lambda t: t.values[1]) # values[1] = MCC

print(f"Best Multi-Objective (F1, MCC): {best_trial.values}")

# Seçilen parametrelerle final model eğitimi
cb = CatBoostClassifier(iterations=250, verbose=0, **best_trial.params)
cb.fit(emb_tr, ytr)

pred = cb.predict(emb_ts)
prob = cb.predict_proba(emb_ts)[:,1]

results.append(metrics(yts, pred, prob, "Chained Hybrid"))
trained_models["Chained Hybrid"] = cb
internal_preds["Chained Hybrid"] = pred
internal_probs["Chained Hybrid"] = prob


# =====================================================
# FINAL
# =====================================================
final=pd.DataFrame(results)
print("\nFINAL TABLE\n")
print(final.round(4))

print("\nUsing fixed global threshold (distribution shift adjusted)")


print("\n==============================")
print("EXTERNAL VALIDATION")
print("==============================")

val_full = pd.read_csv("kp_meropenem_external_validation.csv", sep=";")

# -----------------------------
# METADATA AYIR
# -----------------------------
meta = val_full[["BioProjectAccession","IsolationCountry"]].copy()

# -----------------------------
# MODEL INPUT (metadata çıkar)
# -----------------------------
val = val_full.drop(columns=["BioProjectAccession","IsolationCountry"])

if "Genome ID" in val.columns:
    val = val.drop(columns=["Genome ID"])

y_val = val["Resistant Phenotype"].map({"Susceptible":0,"Resistant":1})
X_val = val.drop(columns=["Resistant Phenotype"])
X_val = X_val[X.columns]
# TRAIN FEATURE ALIGNMENT
X_val = X_val.reindex(columns=X.columns, fill_value=0)

print("External size:", len(X_val))
print("Class dist:\n", y_val.value_counts())

# ===============================
# EN İYİ MODELİ SEÇ
# ===============================

best_idx = final["F1"].idxmax()
best_model_name = final.loc[best_idx, "Model"]
# ===============================
# FEATURE ALIGN
# ===============================
X_val = X_val.reindex(columns=X.columns, fill_value=0)


# ===============================
# FINAL FULL-DATA RETRAIN (Q1 SAFE VERSION)
# ===============================
calibrated_threshold=0.05
best_model = trained_models[best_model_name]

if best_model_name in ["TabTransformer-CatBoost", "Chained Hybrid"]:

    extractor_full = TabTransformerExtractor(X.shape[1]).to(device)
    extractor_full = train_extractor(extractor_full, X, y)

    emb_full = get_emb(extractor_full, X)
    emb_val  = get_emb(extractor_full, X_val)

    final_model = CatBoostClassifier(**best_model.get_params())
    final_model.fit(emb_full, y)

    prob_val = final_model.predict_proba(emb_val)[:,1]

else:
    final_model = CatBoostClassifier(**best_model.get_params())
    final_model.fit(X, y)

    prob_val = final_model.predict_proba(X_val)[:,1]


pred_val = (prob_val > calibrated_threshold).astype(int)

ext_results = metrics(y_val, pred_val, prob_val, "External Validation")
print("\nEXTERNAL RESULTS\n")
print(pd.DataFrame([ext_results]).round(4))


# =====================================================
# PER BIOPROJECT PERFORMANCE
# =====================================================

print("\nPER BIOPROJECT PERFORMANCE")

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

    if best_model_name in ["TabTransformer-CatBoost","Chained Hybrid"]:
        emb_p = get_emb(extractor_full, X_p)
        prob_p = final_model.predict_proba(emb_p)[:,1]
    else:
        prob_p = final_model.predict_proba(X_p)[:,1]

    pred_p = (prob_p > calibrated_threshold).astype(int)
    res = metrics(y_p, pred_p, prob_p, project)
    res["n"] = idx.sum()
    per_project_results.append(res)

per_df = pd.DataFrame(per_project_results)
per_df.to_csv("TABLE_per_project.csv", index=False)
print(per_df.round(4))


# =====================================================
# COUNTRY PERFORMANCE
# =====================================================

print("\nPER COUNTRY PERFORMANCE")

country_results = []

for c in meta["IsolationCountry"].unique():

    idx = meta["IsolationCountry"] == c

    if idx.sum() < 5:
        continue

    X_c = X_val_meta[idx]
    y_c = y_val_meta[idx]

    if best_model_name in ["TabTransformer-CatBoost","Chained Hybrid"]:
        emb_c = get_emb(extractor_full, X_c)
        prob_c = final_model.predict_proba(emb_c)[:,1]
    else:
        prob_c = final_model.predict_proba(X_c)[:,1]

    pred_c = (prob_c > calibrated_threshold).astype(int)
    res = metrics(y_c, pred_c, prob_c, c)
    res["n"] = idx.sum()
    country_results.append(res)

country_df = pd.DataFrame(country_results)
country_df.to_csv("TABLE_country.csv", index=False)
print(country_df.round(4))


# ===============================
# GRAFİKLER
# ===============================
best_idx = final["F1"].idxmax()
best_name = final.loc[best_idx,"Model"]

best_pred = internal_preds[best_name]
best_prob = internal_probs[best_name]

plot_roc(yts, best_prob, best_name)
plot_roc(y_val, prob_val, "External")
# =====================================================
# Q1 FIGURES
# =====================================================

def fig_pipeline():
    plt.figure(figsize=(8,4))
    plt.text(0.1,0.7,"Genomic matrix")
    plt.text(0.35,0.7,"Clade split")
    plt.text(0.6,0.7,"Transformer")
    plt.text(0.85,0.7,"CatBoost")
    plt.arrow(0.2,0.7,0.1,0)
    plt.arrow(0.45,0.7,0.1,0)
    plt.arrow(0.7,0.7,0.1,0)
    plt.axis("off")
    plt.savefig("FIG1_pipeline.png",dpi=300,bbox_inches="tight")
    plt.close()

def fig_pr(y,prob,name):
    p,r,_=precision_recall_curve(y,prob)
    plt.figure(figsize=(5,4))
    plt.plot(r,p)
    plt.title("PR curve")
    plt.savefig(f"FIG3_pr_{name}.png",dpi=300)
    plt.close()

def fig_calibration(y,prob,name):
    frac,mean=calibration_curve(y,prob,n_bins=10)
    plt.figure(figsize=(5,4))
    plt.plot(mean,frac,marker="o")
    plt.plot([0,1],[0,1],"--")
    plt.title("Calibration")
    plt.savefig(f"FIG4_calibration_{name}.png",dpi=300)
    plt.close()

def fig_attention(extractor, X):

    extractor.eval()

    with torch.no_grad():
        x = torch.tensor(X.values[:128], dtype=torch.float32).to(device)
        _, attn_list = extractor(x, return_attn=True)

    # (L, B, H, T, T)
    attn = torch.stack(attn_list)

    # layer average
    attn = attn.mean(0)   # (B, H, T, T)

    # batch average
    attn = attn.mean(0)   # (H, T, T)

    # head average
    attn = attn.mean(0)   # (T, T)

    attn = attn.cpu().numpy()

    plt.figure(figsize=(6,5))
    sns.heatmap(attn, cmap="viridis")
    plt.title("Attention matrix (layer+head averaged)")
    plt.tight_layout()
    plt.savefig("FIG6_attention.png", dpi=300)
    plt.close()

    print("FIG6_attention.png saved")

def fig_clade(groups,y):
    df=pd.DataFrame({"g":groups,"y":y})
    ct=pd.crosstab(df.g,df.y)
    sns.heatmap(ct,cmap="Blues")
    plt.title("Clade leakage check")
    plt.savefig("FIG7_clade.png",dpi=300)
    plt.close()

def fig_external_bar(internal,external):
    df=pd.concat([internal,external])
    sns.barplot(data=df,x="Model",y="F1")
    plt.xticks(rotation=45)
    plt.savefig("FIG8_external.png",dpi=300,bbox_inches="tight")
    plt.close()

def fig_optuna(study,name):
    vals=[t.value for t in study.trials if t.value is not None]
    plt.plot(vals)
    plt.title("Optuna history")
    plt.savefig(f"SUPP_optuna_{name}.png",dpi=300)
    plt.close()

# ===============================
# FEATURE ALIGN
# ===============================
X_val = X_val.reindex(columns=X.columns, fill_value=0)

 
# =====================================================
# GENERATE ALL Q1 FIGURES
# =====================================================

print("\nGenerating Q1 figures...")

fig_pipeline()

fig_pr(yts,best_prob,"internal")
fig_pr(y_val,prob_val,"external")

fig_calibration(yts,best_prob,"internal")
fig_calibration(y_val,prob_val,"external")

# =====================================
# SHAP — EMBEDDING SPACE ONLY
# =====================================

print("\nComputing SHAP on transformer embeddings...")

# sadece hybrid modeller için
if best_model_name in ["TabTransformer-CatBoost", "Chained Hybrid"]:

    shap_X = pd.DataFrame(emb_tr)

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(shap_X)

    shap.summary_plot(
        shap_values,
        shap_X,
        show=False,
        plot_size=(7,5)
    )

    plt.title("Feature importance in transformer embedding space")
    plt.tight_layout()
    plt.savefig("FIG5_shap.png", dpi=600)
    plt.close()

    print("FIG5 SHAP saved (embedding space)")

fig_attention(extractor_full, X)

fig_clade(groups,y)

internal_df = final
external_df = pd.DataFrame([ext_results])
fig_external_bar(internal_df,external_df)

print("All Q1 figures saved.")
   
plt.figure(figsize=(7,5))
sns.countplot(data=meta, y="IsolationCountry")
plt.title("Geographic distribution of external isolates")
plt.tight_layout()
plt.savefig("SUPP_country_distribution.png", dpi=300)
plt.close()


import matplotlib.pyplot as plt

def plot_combined_roc(y_internal, prob_internal,
                      y_external, prob_external,
                      save_name="FIG3_ROC_combined.png"):

    # ROC hesapla
    fpr_int, tpr_int, _ = roc_curve(y_internal, prob_internal)
    auc_int = auc(fpr_int, tpr_int)

    fpr_ext, tpr_ext, _ = roc_curve(y_external, prob_external)
    auc_ext = auc(fpr_ext, tpr_ext)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # -------------------
    # (A) INTERNAL
    # -------------------
    axes[0].plot(fpr_int, tpr_int, linewidth=2)
    axes[0].plot([0,1], [0,1], linestyle="--")
    axes[0].set_title("(A) Internal Test ROC\nChained Hybrid")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend([f"AUC = {auc_int:.3f}"])

    # -------------------
    # (B) EXTERNAL
    # -------------------
    axes[1].plot(fpr_ext, tpr_ext, linewidth=2)
    axes[1].plot([0,1], [0,1], linestyle="--")
    axes[1].set_title("(B) External Validation ROC\nChained Hybrid")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend([f"AUC = {auc_ext:.3f}"])

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()

    print("ROC figure saved:", save_name)

plot_combined_roc(
    yts, best_prob,       # internal
    y_val, prob_val       # external
)    

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne_dual(
        emb_test, y_test,
        emb_ext, y_ext,
        save_name="FIG2_TSNE.png"):

    # ---------------- TEST TSNE ----------------
    tsne_test = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )
    z_test = tsne_test.fit_transform(emb_test)

    # ---------------- EXTERNAL TSNE ----------------
    tsne_ext = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )
    z_ext = tsne_ext.fit_transform(emb_ext)

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    # -------- A TEST ----------
    axes[0].scatter(
        z_test[:,0], z_test[:,1],
        c=y_test,
        s=20
    )
    axes[0].set_title("(A) Internal test embeddings\nChained Hybrid")
    axes[0].set_xlabel("t-SNE1")
    axes[0].set_ylabel("t-SNE2")

    # -------- B EXTERNAL ----------
    axes[1].scatter(
        z_ext[:,0], z_ext[:,1],
        c=y_ext,
        s=20
    )
    axes[1].set_title("(B) External embeddings\nChained Hybrid")
    axes[1].set_xlabel("t-SNE1")
    axes[1].set_ylabel("t-SNE2")

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()

    print("t-SNE figure saved:", save_name)


# ÇAĞRI
plot_tsne_dual(
    emb_ts, yts,     # internal TEST
    emb_val, y_val   # external
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ==============================
# ATTENTION MATRIX
# ==============================
def get_attention_matrix(extractor, X, device):

    extractor.eval()

    with torch.no_grad():
        x = torch.tensor(X.values[:128], dtype=torch.float32).to(device)
        _, attn_list = extractor(x, return_attn=True)

    # (L, B, H, T, T)
    attn = torch.stack(attn_list)

    attn = attn.mean(0)   # layer avg → (B,H,T,T)
    attn = attn.mean(0)   # batch avg → (H,T,T)
    attn = attn.mean(0)   # head avg → (T,T)

    return attn.cpu().numpy()

# ==============================
# TOP PAIRS
# ==============================
def extract_top_pairs(att_matrix, gene_names, top_k=20):

    pairs = []

    for i in range(len(gene_names)):
        for j in range(i+1, len(gene_names)):
            pairs.append((gene_names[i], gene_names[j], att_matrix[i,j]))

    df = pd.DataFrame(pairs, columns=["Gene_A","Gene_B","Attention"])
    df = df.sort_values("Attention", ascending=False).head(top_k)

    return df

# ==============================
# Q1 PLOT
# ==============================
def plot_attention_bar(df):

    labels = df["Gene_A"] + " ↔ " + df["Gene_B"]

    plt.figure(figsize=(5.5,4))
    plt.barh(labels[::-1], df["Attention"][::-1])
    plt.xlabel("Attention weight")
    plt.title("Top genomic interactions")
    plt.tight_layout()
    plt.savefig("FIG6_attention_pairs.png", dpi=600)
    plt.close()

    print("FIG6 saved")

att = get_attention_matrix(extractor_full, X, device)
genes = X.columns.tolist()

top_df = extract_top_pairs(att, genes, top_k=20)
plot_attention_bar(top_df)

# =====================================================
# UNIQUE GENOME COUNT (EXTERNAL)  — FOR PAPER REPORTING
# =====================================================

print("\n==============================")
print("UNIQUE EXTERNAL GENOME COUNTS")
print("==============================")

# --- external dataframe tekrar oluştur ---
val_full = pd.read_csv("validationPRJlison.csv", sep=";")

# metadata ayır
meta_tmp = val_full[["BioProjectAccession","IsolationCountry"]].copy()

val_tmp = val_full.drop(columns=["BioProjectAccession","IsolationCountry"])

if "Genome ID" in val_tmp.columns:
    val_tmp = val_tmp.drop(columns=["Genome ID"])

y_tmp = val_tmp["Resistant Phenotype"].map({"Susceptible":0,"Resistant":1})
X_tmp = val_tmp.drop(columns=["Resistant Phenotype"])

# TRAIN FEATURE ALIGN
X_tmp = X_tmp.reindex(columns=X.columns, fill_value=0)

# =====================================================
# SAFE DEDUP (EXTERNAL)
# =====================================================

genome_sig = X_tmp.astype(str).agg("_".join, axis=1)

df_ext = pd.concat([X_tmp, y_tmp], axis=1)
df_ext["sig"] = genome_sig

label_counts = df_ext.groupby("sig")["Resistant Phenotype"].nunique()
valid_sigs = label_counts[label_counts == 1].index

df_ext_clean = (
    df_ext[df_ext["sig"].isin(valid_sigs)]
    .drop_duplicates(subset="sig")
    .drop(columns="sig")
    .reset_index(drop=True)
)

y_unique = df_ext_clean["Resistant Phenotype"]

print("Total external isolates:", len(val_tmp))
print("Unique genomes:", len(df_ext_clean))

print("\nUnique class distribution:")
print(y_unique.value_counts().rename({0:"Susceptible",1:"Resistant"}))
print("External duplicate genomes with conflict:")
print(label_counts[label_counts>1].head())
print(val_full.groupby(genome_sig)["Resistant Phenotype"].nunique().value_counts())

import numpy as np
import pandas as pd

def map_latent_to_genes(X_df, embeddings, target_dimensions, top_k=10):
    """
    Latent boyutlar ile orijinal genler arasındaki korelasyonu hesaplar.
    """
    gene_names = X_df.columns.tolist()
    X_matrix = X_df.values
    results = []

    for d in target_dimensions:
        dim_vector = embeddings[:, d]
        correlations = []
        
        for i, gene in enumerate(gene_names):
            # Pearson korelasyon katsayısı
            corr = np.corrcoef(X_matrix[:, i], dim_vector)[0, 1]
            if not np.isnan(corr):
                correlations.append((gene, corr))
        
        # Korelasyon gücüne göre sırala (mutlak değerce büyükten küçüğe)
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for gene, corr in correlations[:top_k]:
            results.append({
                "Latent_Dimension": d,
                "Gene_Name": gene,
                "Correlation": round(corr, 4),
                "Direction": "Positive" if corr > 0 else "Negative"
            })
            
    return pd.DataFrame(results)

# --- UYGULAMA ---
# SHAP grafiğinde en etkili bulduğunuz boyutlar
target_dims = [11, 18, 30, 2,23, 29]

# Eğitim setindeki (xtr) genler ile elde edilen embeddingler (emb_tr) arasındaki ilişki
mapping_df = map_latent_to_genes(xtr, emb_tr, target_dims)

# Sonuçları ekrana bas ve CSV olarak kaydet
print("\n=== LATENT DIMENSION TO GENE MAPPING (Top 10 Genes per Dim) ===")
print(mapping_df)
mapping_df.to_csv("TABLE_latent_gene_mapping.csv", index=False)

# Özet Yorum Oluşturma (Discussion için)
for d in target_dims:
    top_gene = mapping_df[mapping_df["Latent_Dimension"] == d].iloc[0]["Gene_Name"]
    corr_val = mapping_df[mapping_df["Latent_Dimension"] == d].iloc[0]["Correlation"]
    print(f"Dimension {d} is primarily driven by {top_gene} (r={corr_val})")

# Tüm boyutları analiz eden versiyon
target_dims = list(range(32)) # 0'dan 31'e kadar tüm boyutlar

mapping_df = map_latent_to_genes(xtr, emb_tr, target_dims, top_k=5) # Her boyut için en iyi 5 geni getir

# Tam listeyi kaydet
mapping_df.to_csv("FULL_LATENT_GENE_MAPPING_32D.csv", index=False)

print("32 boyutun tamamı analiz edildi ve FULL_LATENT_GENE_MAPPING_32D.csv dosyasına kaydedildi.")    

corr_matrix = np.corrcoef(emb_tr.T)
plt.figure(figsize=(10, 8))  # ← bunu ekleyin
sns.heatmap(corr_matrix, cmap="coolwarm")  
plt.tight_layout()
plt.savefig("FIG7_correlation.png", dpi=600, bbox_inches="tight")
plt.close()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_cm_dual(y_int, pred_int, y_ext, pred_ext):

    cm_int = confusion_matrix(y_int, pred_int).astype(float)
    cm_ext = confusion_matrix(y_ext, pred_ext).astype(float)

    cm_int = cm_int / cm_int.sum(axis=1)[:,None]
    cm_ext = cm_ext / cm_ext.sum(axis=1)[:,None]

    fig, axes = plt.subplots(1,2, figsize=(8,4))

    # INTERNAL
    sns.heatmap(
        cm_int,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=["Susceptible","Resistant"],
        yticklabels=["Susceptible","Resistant"],
        ax=axes[0]
    )
    axes[0].set_title("(A) Internal test")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # EXTERNAL
    sns.heatmap(
        cm_ext,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=["Susceptible","Resistant"],
        yticklabels=["Susceptible","Resistant"],
        ax=axes[1]
    )
    axes[1].set_title("(B) External validation")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig("FIG_CM_combined.png", dpi=600)
    plt.close()

    print("FIG_CM_combined.png saved")

plot_cm_dual(yts, best_pred, y_val, pred_val)

# =====================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =====================================================
from sklearn.utils import resample

def bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=1000, random_state=42):
    np.random.seed(random_state)

    mcc_scores = []
    auc_scores = []
    f1_scores = []

    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = resample(range(n))
        y_t = y_true[idx]
        y_p = y_pred[idx]
        y_pr = y_prob[idx]

        # AUROC hesaplanabilmesi için iki sınıf da olmalı
        if len(np.unique(y_t)) < 2:
            continue

        mcc_scores.append(matthews_corrcoef(y_t, y_p))
        auc_scores.append(roc_auc_score(y_t, y_pr))
        f1_scores.append(f1_score(y_t, y_p))

    def summary(scores):
        mean = np.mean(scores)
        lower = np.percentile(scores, 2.5)
        upper = np.percentile(scores, 97.5)
        return mean, lower, upper

    return {
        "MCC": summary(mcc_scores),
        "AUROC": summary(auc_scores),
        "F1": summary(f1_scores)
    }

print("\n==============================")
print("BOOTSTRAP CONFIDENCE INTERVALS")
print("==============================")

# ---------- INTERNAL ----------
print("\nINTERNAL TEST SET")

internal_ci = bootstrap_ci(
    yts.values if hasattr(yts, "values") else yts,
    best_pred,
    best_prob,
    n_bootstrap=1000
)

for metric, values in internal_ci.items():
    mean, lower, upper = values
    print(f"{metric}: {mean:.4f} (95% CI: {lower:.4f} – {upper:.4f})")


# ---------- EXTERNAL ----------
print("\nEXTERNAL VALIDATION SET")

external_ci = bootstrap_ci(
    y_val.values if hasattr(y_val, "values") else y_val,
    pred_val,
    prob_val,
    n_bootstrap=1000
)

for metric, values in external_ci.items():
    mean, lower, upper = values
    print(f"{metric}: {mean:.4f} (95% CI: {lower:.4f} – {upper:.4f})")

# =====================================================
# RAW DATA BASELINE CATBOOST (NO DEDUP / NO CLADE SPLIT)
# =====================================================

print("\n==============================")
print("RAW DATA BASELINE CATBOOST")
print("==============================")

# ---- HAM DATA YÜKLE ----
df_raw = pd.read_csv("asilverisetigenler.csv", sep=";")

if "Genome ID" in df_raw.columns:
    df_raw = df_raw.drop(columns=["Genome ID"])

y_raw = df_raw["Resistant Phenotype"].map({"Susceptible":0,"Resistant":1})
X_raw = df_raw.drop(columns=["Resistant Phenotype"])

# sabit sütun sil
X_raw = X_raw.loc[:, X_raw.nunique() > 1]

# klasik random split (clade-aware değil)
from sklearn.model_selection import train_test_split
Xr_tr, Xr_ts, yr_tr, yr_ts = train_test_split(
    X_raw, y_raw,
    test_size=0.2,
    stratify=y_raw,
    random_state=42
)

# ---- SAME CATBOOST PARAMETERS ----
raw_cb = CatBoostClassifier(
    iterations=250,
    depth=8,
    verbose=0,
    random_state=42
)

raw_cb.fit(Xr_tr, yr_tr)

raw_pred = raw_cb.predict(Xr_ts)
raw_prob = raw_cb.predict_proba(Xr_ts)[:,1]

# ---- METRICS ----
raw_metrics = metrics(yr_ts, raw_pred, raw_prob, "Raw CatBoost (No Dedup)")
print(pd.DataFrame([raw_metrics]).round(4))

# ---- BOOTSTRAP CI ----
print("\nRAW DATA BOOTSTRAP CI")

raw_ci = bootstrap_ci(
    yr_ts.values if hasattr(yr_ts, "values") else yr_ts,
    raw_pred,
    raw_prob,
    n_bootstrap=1000
)

for metric, values in raw_ci.items():
    mean, lower, upper = values
    print(f"{metric}: {mean:.4f} (95% CI: {lower:.4f} – {upper:.4f})")    

print("\n==============================")
print("RAW MODEL – EXTERNAL VALIDATION")
print("==============================")

# --- VALIDATION DATA LOAD ---
val_raw = pd.read_csv("validationPRJlison.csv", sep=";")

if "Genome ID" in val_raw.columns:
    val_raw = val_raw.drop(columns=["Genome ID"])

# metadata varsa çıkar
for col in ["BioProjectAccession", "IsolationCountry"]:
    if col in val_raw.columns:
        val_raw = val_raw.drop(columns=[col])

y_val_raw = val_raw["Resistant Phenotype"].map({"Susceptible":0,"Resistant":1})
X_val_raw = val_raw.drop(columns=["Resistant Phenotype"])

# ---- FEATURE ALIGNMENT ----
# Raw model X_raw üzerinde eğitildiği için aynı feature setini zorla
X_val_raw = X_val_raw.reindex(columns=X_raw.columns, fill_value=0)

# ---- PREDICTION ----
raw_ext_pred = raw_cb.predict(X_val_raw)
raw_ext_prob = raw_cb.predict_proba(X_val_raw)[:,1]

# ---- METRICS ----
raw_ext_metrics = metrics(
    y_val_raw,
    raw_ext_pred,
    raw_ext_prob,
    "Raw CatBoost External"
)

print(pd.DataFrame([raw_ext_metrics]).round(4))

# ---- BOOTSTRAP CI ----
print("\nRAW EXTERNAL BOOTSTRAP CI")

raw_ext_ci = bootstrap_ci(
    y_val_raw.values if hasattr(y_val_raw, "values") else y_val_raw,
    raw_ext_pred,
    raw_ext_prob,
    n_bootstrap=1000
)

for metric, values in raw_ext_ci.items():
    mean, lower, upper = values
    print(f"{metric}: {mean:.4f} (95% CI: {lower:.4f} – {upper:.4f})")    

# Ham veriyi okuduktan hemen sonra:
print(f"Toplam Ham İzolat Sayısı: {len(df_raw)}")
print(f"X_raw Satır Sayısı: {X_raw.shape[0]}")
   
# =====================================================
# DELONG TEST (AUROC COMPARISON)
# =====================================================
from scipy import stats

def delong_test(y_true, prob1, prob2, n_boot=1000):
    auc_diffs = []

    y_true = np.array(y_true)

    for _ in range(n_boot):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)

        y_b = y_true[idx]
        p1 = prob1[idx]
        p2 = prob2[idx]

        if len(np.unique(y_b)) < 2:
            continue

        auc1 = roc_auc_score(y_b, p1)
        auc2 = roc_auc_score(y_b, p2)

        auc_diffs.append(auc1 - auc2)

    auc_diffs = np.array(auc_diffs)

    p_value = 2 * min(
        np.mean(auc_diffs <= 0),
        np.mean(auc_diffs >= 0)
    )

    return p_value


print("\n==============================")
print("DELONG TEST (AUROC COMPARISON)")
print("==============================")

y_true = yts.values if hasattr(yts, "values") else yts

prob_baseline = internal_probs["Baseline CatBoost"]
prob_hybrid   = internal_probs["Chained Hybrid"]

p_val = delong_test(y_true, prob_baseline, prob_hybrid)

print("Baseline AUROC:", roc_auc_score(y_true, prob_baseline))
print("Hybrid AUROC:", roc_auc_score(y_true, prob_hybrid))
print("DeLong p-value:", p_val)
# =====================================================
# REVIEWER 1 – COMMENT 3: THRESHOLD SENSITIVITY ANALYSIS
# External validation only — distribution shift justification
# =====================================================
thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
sensitivity_results = []

for t in thresholds:
    pred_t = (prob_val > t).astype(int)
    if len(np.unique(pred_t)) < 2:
        continue
    sensitivity_results.append({
        "Threshold": t,
        "Accuracy":  accuracy_score(y_val, pred_t),
        "Precision": precision_score(y_val, pred_t, zero_division=0),
        "Recall":    recall_score(y_val, pred_t, zero_division=0),
        "F1":        f1_score(y_val, pred_t, zero_division=0),
        "MCC":       matthews_corrcoef(y_val, pred_t)
    })

sens_df = pd.DataFrame(sensitivity_results).round(4)
print(sens_df.to_string(index=False))
sens_df.to_csv("TABLE_threshold_sensitivity_external.csv", index=False)
print("TABLE_threshold_sensitivity_external.csv saved")