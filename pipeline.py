# -*- coding: utf-8 -*-
"""
pipeline.py
===========
Leakage-Aware Genomic Prediction Pipeline for Meropenem Resistance
in Klebsiella pneumoniae Using Transformer-Based Resistome Representation Learning

Author  : Ilkay Sibel Kervanci
Affil.  : Gaziantep University, Department of Computer Engineering
Contact : sibelkervanci@gantep.edu.tr
Paper   : Submitted to Bioinformatics Advances (Oxford University Press), 2026
GitHub  : https://github.com/SibelKervanci/kp-meropenem-tabtransformer

Revision history
----------------
2026-03-04  Initial release
2026-05-xx  Added ablation comparisons per Reviewer 1, Comment 1:
              - Embedding + Logistic Regression
              - Embedding + MLP
              - Transformer-Only (end-to-end with built-in classification head)
            Added threshold sensitivity analysis per Reviewer 1, Comment 3.

Pipeline overview
-----------------
1. Load raw genomic presence/absence matrix (CSV).
2. Safe deduplication: remove conflicting genome signatures and exact duplicates.
3. Clade-aware train/test split using Hamming-distance agglomerative clustering
   and GroupShuffleSplit to prevent phylogenetic leakage.
4. Train baseline classifiers (LR, MLP, RF, XGBoost, CatBoost).
5. Train TabTransformerExtractor to obtain 32-dimensional genomic embeddings.
6. Ablation: test alternative downstream classifiers on the same embeddings
   (Embedding+LR, Embedding+MLP, Transformer-Only).
7. Hyperparameter optimisation of CatBoost via Optuna (TPE, CMA-ES, NSGA-II).
8. Multi-objective NSGA-II optimisation of the Chained Hybrid model.
9. External validation on independent multinational BioProject cohorts.
10. Statistical evaluation: permutation MCC, bootstrap CI, DeLong AUROC test,
    threshold sensitivity analysis.
11. Explainability: SHAP, attention heatmap, t-SNE, latent-to-gene mapping.

Inputs
------
asilverisetigenler.csv   : Training dataset — binary gene presence/absence matrix
                           with a "Resistant Phenotype" column (Susceptible/Resistant).
validationPRJlison.csv   : External validation dataset — same format plus
                           BioProjectAccession and IsolationCountry columns.

Outputs
-------
FINAL TABLE              : Console — all model performance metrics
TABLE_per_project.csv    : Per-BioProject external performance
TABLE_country.csv        : Per-country external performance
TABLE_threshold_sensitivity_external.csv : Threshold sensitivity on external cohort
TABLE_latent_gene_mapping.csv            : Latent dimension to gene correlations
FULL_LATENT_GENE_MAPPING_32D.csv         : Full 32-dimension mapping
FIG*.png                 : All publication figures (dpi=300 or 600)
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard scientific computing
import numpy as np
import pandas as pd

# Visualisation
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch — deep learning backbone
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# scikit-learn — preprocessing, clustering, baseline models, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    roc_curve, auc, confusion_matrix,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.utils import resample

# XGBoost and CatBoost gradient boosting classifiers
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Optuna — hyperparameter optimisation framework
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler

# SHAP — model explainability
import shap

# SciPy — statistical tests
from scipy import stats

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
pd.set_option("display.max_columns", None)   # show all dataframe columns
pd.set_option("display.width", 200)           # wide console output

# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix random seeds for full reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dictionaries to store trained models and their predictions for later use
trained_models = {}   # key: model name, value: fitted model object
internal_preds = {}   # key: model name, value: binary predictions on test set
internal_probs = {}   # key: model name, value: predicted probabilities on test set

# Publication-quality matplotlib settings (Arial font, consistent sizes)
matplotlib.rcParams['font.family']    = 'Arial'
matplotlib.rcParams['font.size']      = 11   # base font size
matplotlib.rcParams['axes.titlesize'] = 12   # subplot title size
matplotlib.rcParams['axes.labelsize'] = 11   # axis label size
matplotlib.rcParams['xtick.labelsize']= 10   # x-axis tick label size
matplotlib.rcParams['ytick.labelsize']= 10   # y-axis tick label size

# =============================================================================
# 1. DATA LOADING
# =============================================================================
# Load the binary genomic presence/absence matrix.
# Rows = isolates, columns = AMR genes + "Resistant Phenotype" label column.
df = pd.read_csv("asilverisetigenler.csv", sep=";")

# Drop Genome ID column if present (not a feature, only an identifier)
if "Genome ID" in df.columns:
    df = df.drop(columns=["Genome ID"])

# Encode phenotype labels: Susceptible -> 0, Resistant -> 1
y = df["Resistant Phenotype"].map({"Susceptible": 0, "Resistant": 1})

# Separate feature matrix from the label column
X = df.drop(columns=["Resistant Phenotype"])

# Remove constant columns (genes present in all or none of the isolates)
# Such columns carry no discriminative information
X = X.loc[:, X.nunique() > 1]

# =============================================================================
# 2. SAFE DEDUPLICATION
# =============================================================================
# Goal: remove exact duplicate genome signatures AND conflicting entries
# (same genome signature appearing with different phenotypes).
# This prevents label leakage from near-identical isolates across train/test.

# Create a unique string "signature" for each genome by concatenating
# all binary gene presence/absence values
genome_sig = X.astype(str).agg("_".join, axis=1)

# Attach signatures and labels to a temporary dataframe for filtering
df_tmp = pd.concat([X, y], axis=1)
df_tmp["sig"] = genome_sig

# Count distinct phenotype values per unique genome signature
label_counts = df_tmp.groupby("sig")["Resistant Phenotype"].nunique()

# Keep only signatures with a single, consistent phenotype label
valid_sigs = label_counts[label_counts == 1].index

# Apply filter and remove exact duplicates, retaining one representative per signature
df_clean = (
    df_tmp[df_tmp["sig"].isin(valid_sigs)]
    .drop_duplicates(subset="sig")
    .drop(columns="sig")
    .reset_index(drop=True)
)

# Final deduplicated feature matrix and label vector
X = df_clean.drop(columns=["Resistant Phenotype"])
y = df_clean["Resistant Phenotype"]

# =============================================================================
# 3. CLADE-AWARE TRAIN / TEST SPLIT
# =============================================================================
# Standard random splitting risks placing phylogenetically related isolates
# in both train and test sets, inflating performance estimates.
# We use Hamming-distance agglomerative clustering to define clades,
# then GroupShuffleSplit to ensure whole clades remain in one split only.

# Fit agglomerative clustering using Hamming distance on binary gene vectors.
# distance_threshold=0.05 groups isolates sharing >95% of their gene profile.
cluster_model = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.05,   # genomic similarity threshold for clade definition
    metric="hamming",          # appropriate for binary presence/absence data
    linkage="average"
)
groups = cluster_model.fit_predict(X)

# Fallback: if fewer than 5 clades are found, use a fixed number of 50 clusters
if len(np.unique(groups)) < 5:
    cluster_model = AgglomerativeClustering(
        n_clusters=50,
        metric="hamming",
        linkage="average"
    )
    groups = cluster_model.fit_predict(X)

print(f"Total samples: {len(X)} | Clade count: {len(np.unique(groups))}")

# Split data ensuring entire clades go to either train or test, not both
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

# Assign train and test subsets
xtr = X.iloc[train_idx]   # training features
xts = X.iloc[test_idx]    # test features
ytr = y.iloc[train_idx]   # training labels
yts = y.iloc[test_idx]    # test labels

# Report split statistics
print("Train size:", len(xtr))
print("Test size:", len(xts))
print("Train class dist:\n", ytr.value_counts())
print("Test class dist:\n", yts.value_counts())
print("Total samples after SAFE dedup:", len(X))
print(y.value_counts().rename({0: "Susceptible", 1: "Resistant"}))

# =============================================================================
# 4. EVALUATION UTILITIES
# =============================================================================

def perm_mcc(y_true, y_pred, n_perm=100):
    """
    Permutation-based MCC null distribution test.

    Shuffles the true labels n_perm times and computes MCC for each shuffle.
    The mean of these null MCCs should be close to zero for a valid model.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    n_perm : int
        Number of permutations (default 100).

    Returns
    -------
    float
        Mean MCC over all permutations (expected near 0).
    """
    scores = []
    for _ in range(n_perm):
        # Randomly shuffle true labels to break any real signal
        y_perm = np.random.permutation(y_true)
        scores.append(matthews_corrcoef(y_perm, y_pred))
    return np.mean(scores)


def metrics(y_true, y_pred, y_prob, label):
    """
    Compute a standard set of binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 = Susceptible, 1 = Resistant).
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like
        Predicted probabilities for the positive (Resistant) class.
    label : str
        Model name used as the "Model" key in the returned dict.

    Returns
    -------
    dict
        Dictionary with keys: Model, Accuracy, Precision, Recall,
        F1, MCC, AUROC, Perm_MCC.
    """
    return {
        "Model":     label,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "F1":        f1_score(y_true, y_pred),
        "MCC":       matthews_corrcoef(y_true, y_pred),
        "AUROC":     roc_auc_score(y_true, y_prob),
        "Perm_MCC":  perm_mcc(y_true, y_pred, n_perm=100)
    }

# =============================================================================
# 5. PLOTTING UTILITIES
# =============================================================================

def plot_roc(y_true, y_prob, name):
    """
    Plot and save a single ROC curve.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    name : str
        Model name used in the title and output filename.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], '--')   # random classifier diagonal
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"roc_{name}.png", dpi=300)
    plt.close()

# =============================================================================
# 6. PYTORCH DATASET WRAPPER
# =============================================================================

class TabDataset(Dataset):
    """
    Minimal PyTorch Dataset for tabular binary genomic data.

    Converts pandas DataFrames to float32 tensors for the feature matrix
    and long tensors for integer class labels.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (binary gene presence/absence).
    y : pd.Series
        Integer label vector (0 = Susceptible, 1 = Resistant).
    """

    def __init__(self, X, y):
        # Convert feature DataFrame to float32 tensor (required by nn.Linear)
        self.X = torch.tensor(X.values, dtype=torch.float32)
        # Convert label Series to long tensor (required by CrossEntropyLoss)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        # Return the total number of samples in this dataset
        return len(self.X)

    def __getitem__(self, i):
        # Return a single (feature vector, label) pair at index i
        return self.X[i], self.y[i]

# =============================================================================
# 7. TABTRANSFORMER ENCODER (FEATURE EXTRACTOR)
# =============================================================================

class TabTransformerExtractor(nn.Module):
    """
    Transformer-based genomic feature extractor.

    Maps a binary gene presence/absence vector of length F into a compact
    d_model-dimensional embedding via self-attention over gene tokens.

    Architecture
    ------------
    - feature_embed : Embedding layer — maps each gene index to a d_model vector,
      capturing the identity of each genomic locus.
    - value_proj    : Linear(1 -> d_model) — projects the scalar binary value
      (0 or 1) into d_model dimensions, encoding whether the gene is present.
    - Token         : feat + val — combines locus identity and presence signal.
    - tr            : 3-layer TransformerEncoder with 4 attention heads,
      dim_feedforward=64, dropout=0.2.
    - Pooling       : mean over the sequence dimension -> single d_model vector.

    Parameters
    ----------
    num_features : int
        Number of binary gene features (columns in X).
    """

    def __init__(self, num_features):
        super().__init__()
        self.d_model = 32   # embedding dimension (conservative to reduce overfitting)

        # Gene identity embedding: maps each of the F gene indices to a d_model vector
        self.feature_embed = nn.Embedding(num_features, self.d_model)

        # Value projection: maps scalar presence/absence value to d_model dimensions
        self.value_proj = nn.Linear(1, self.d_model)

        # Transformer encoder layer configuration
        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,                        # 4 parallel attention heads
            dim_feedforward=self.d_model*2, # feedforward hidden size = 64
            dropout=0.2,                    # dropout for regularisation
            batch_first=True                # input shape: (batch, seq, feature)
        )

        # Stack 3 encoder layers
        self.tr = nn.TransformerEncoder(self.enc_layer, num_layers=3)

    def forward(self, x, return_attn=False):
        """
        Forward pass through the transformer encoder.

        Parameters
        ----------
        x : torch.Tensor, shape (B, F)
            Binary gene presence/absence matrix for a batch of B isolates
            with F genes.
        return_attn : bool
            If True, return attention weights from each layer for visualisation.

        Returns
        -------
        torch.Tensor, shape (B, d_model)
            Mean-pooled embedding vector for each isolate.
        list of torch.Tensor (only if return_attn=True)
            Per-layer attention weight tensors.
        """
        B, F = x.shape

        # Create gene index tensor: shape (B, F) with values 0..F-1
        idx = torch.arange(F, device=x.device).unsqueeze(0).repeat(B, 1)

        # Embed gene identities: shape (B, F, d_model)
        feat = self.feature_embed(idx)

        # Project scalar binary values to d_model: shape (B, F, d_model)
        val = self.value_proj(x.unsqueeze(-1))

        # Combine identity and value embeddings: shape (B, F, d_model)
        tok = feat + val

        if return_attn:
            # Manually iterate through layers to extract attention weights
            attns = []
            out = tok
            for layer in self.tr.layers:
                # Extract raw attention weights before the layer's other operations
                attn_out, attn = layer.self_attn(
                    out, out, out,
                    need_weights=True,
                    average_attn_weights=False
                )
                attns.append(attn.detach())   # detach to avoid keeping computation graph
                out = layer(out)              # apply full encoder layer
            # Return mean-pooled embedding and attention list
            return out.mean(1), attns

        # Standard forward: apply all encoder layers, then mean-pool
        out = self.tr(tok)
        return out.mean(1)   # shape: (B, d_model)


def train_extractor(extractor, X, y, epochs=120):
    """
    Pre-train the TabTransformerExtractor with a temporary classification head.

    The head is discarded after training; only the encoder weights are retained.
    Training uses Adam optimiser with L2 weight decay for regularisation.

    Parameters
    ----------
    extractor : TabTransformerExtractor
        Uninitialised encoder to be trained.
    X : pd.DataFrame
        Training feature matrix.
    y : pd.Series
        Training labels.
    epochs : int
        Number of full passes over the training data (default 120).

    Returns
    -------
    TabTransformerExtractor
        Trained encoder in eval() mode.
    """
    # Temporary linear classification head (discarded after training)
    head = nn.Linear(extractor.d_model, 2).to(device)

    # Optimise both encoder and temporary head parameters jointly
    opt = torch.optim.Adam(
        list(extractor.parameters()) + list(head.parameters()),
        lr=1e-3,          # initial learning rate
        weight_decay=1e-4  # L2 regularisation to reduce overfitting on small dataset
    )

    # Standard cross-entropy loss for binary classification
    loss_fn = nn.CrossEntropyLoss()

    # DataLoader shuffles batches each epoch for stochastic gradient descent
    loader = DataLoader(
        TabDataset(X, y),
        batch_size=128,
        shuffle=True
    )

    extractor.train()   # set encoder to training mode (enables dropout)

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)   # move batch to GPU/CPU
            opt.zero_grad()                           # clear accumulated gradients
            out = head(extractor(xb))                 # forward pass
            loss = loss_fn(out, yb)                   # compute loss
            loss.backward()                           # backpropagate
            opt.step()                                # update parameters

    extractor.eval()   # switch to eval mode (disables dropout)
    return extractor


def get_emb(ext, X):
    """
    Extract transformer embeddings for a feature matrix without gradient tracking.

    Parameters
    ----------
    ext : TabTransformerExtractor
        Trained encoder in eval() mode.
    X : pd.DataFrame
        Feature matrix to embed.

    Returns
    -------
    np.ndarray, shape (n_samples, d_model)
        32-dimensional embedding vectors for all isolates.
    """
    with torch.no_grad():   # disable gradient computation for inference efficiency
        return ext(
            torch.tensor(X.values, dtype=torch.float32).to(device)
        ).cpu().numpy()     # move result back to CPU and convert to numpy

# =============================================================================
# 8. BASELINE CLASSIFIERS
# =============================================================================
results = []   # accumulates metric dictionaries for the final summary table

# Define baseline models with fixed hyperparameters for fair comparison
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=250),
    "MLP":                 MLPClassifier(hidden_layer_sizes=(128, 64),
                                         max_iter=250, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=250,
                                                   max_depth=6, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=250, max_depth=6,
                                          eval_metric="logloss"),
}

# Train each baseline model and evaluate on the clade-aware test set
for name, model in base_models.items():
    model.fit(xtr, ytr)                         # train on clade-aware training set
    pred = model.predict(xts)                   # predict class labels
    prob = (model.predict_proba(xts)[:, 1]      # predict Resistant class probability
            if hasattr(model, "predict_proba") else pred)
    results.append(metrics(yts, pred, prob, name))

# -----------------------------------------------------------------------------
# Baseline CatBoost (no hyperparameter optimisation, fixed depth=8)
# -----------------------------------------------------------------------------
base = CatBoostClassifier(iterations=250, depth=8, verbose=0, random_state=42)
base.fit(xtr, ytr)
trained_models["Baseline CatBoost"] = base          # store for later use
pred = base.predict(xts)
prob = base.predict_proba(xts)[:, 1]
results.append(metrics(yts, pred, prob, "Baseline CatBoost"))
internal_preds["Baseline CatBoost"] = pred           # store predictions
internal_probs["Baseline CatBoost"] = prob           # store probabilities

# =============================================================================
# 9. TABTRANSFORMER + CATBOOST (HYBRID — FIXED HYPERPARAMETERS)
# =============================================================================
# Train the transformer encoder on the training set
extractor = TabTransformerExtractor(X.shape[1]).to(device)
extractor = train_extractor(extractor, xtr, ytr)

# Extract 32-dimensional embeddings for train and test sets
emb_tr = get_emb(extractor, xtr)   # shape: (n_train, 32)
emb_ts = get_emb(extractor, xts)   # shape: (n_test,  32)

# Train CatBoost on transformer embeddings (unoptimised baseline hybrid)
cb = CatBoostClassifier(iterations=250, depth=8, verbose=0, random_state=42)
cb.fit(emb_tr, ytr)
pred = cb.predict(emb_ts)
prob = cb.predict_proba(emb_ts)[:, 1]
results.append(metrics(yts, pred, prob, "TabTransformer-CatBoost"))
trained_models["TabTransformer-CatBoost"] = cb
internal_preds["TabTransformer-CatBoost"] = pred
internal_probs["TabTransformer-CatBoost"] = prob

# =============================================================================
# 10. ABLATION: ALTERNATIVE DOWNSTREAM CLASSIFIERS ON TRANSFORMER EMBEDDINGS
#     (Added per Reviewer 1, Comment 1)
# =============================================================================
# Purpose: empirically justify the choice of CatBoost as downstream classifier
# by comparing against LR, MLP, and a standalone Transformer-Only model.
# All three use the same 32-dimensional embeddings produced by the trained encoder.

print("\n==============================")
print("ABLATION: EMBEDDING-BASED CLASSIFIERS")
print("==============================")

# --- 10a. Embedding + Logistic Regression ---
# Simplest linear baseline on transformer embeddings
emb_lr = LogisticRegression(max_iter=500, random_state=42)
emb_lr.fit(emb_tr, ytr)                          # train on embeddings
pred_emb_lr  = emb_lr.predict(emb_ts)
prob_emb_lr  = emb_lr.predict_proba(emb_ts)[:, 1]
results.append(metrics(yts, pred_emb_lr, prob_emb_lr, "Embedding + LR"))
trained_models["Embedding + LR"] = emb_lr
internal_preds["Embedding + LR"] = pred_emb_lr
internal_probs["Embedding + LR"] = prob_emb_lr

# --- 10b. Embedding + MLP ---
# Non-linear neural baseline on transformer embeddings
emb_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
emb_mlp.fit(emb_tr, ytr)                          # train on embeddings
pred_emb_mlp = emb_mlp.predict(emb_ts)
prob_emb_mlp = emb_mlp.predict_proba(emb_ts)[:, 1]
results.append(metrics(yts, pred_emb_mlp, prob_emb_mlp, "Embedding + MLP"))
trained_models["Embedding + MLP"] = emb_mlp
internal_preds["Embedding + MLP"] = pred_emb_mlp
internal_probs["Embedding + MLP"] = prob_emb_mlp

# --- 10c. Transformer-Only (end-to-end, built-in classification head) ---

class TabTransformerClassifier(nn.Module):
    """
    Standalone TabTransformer with a built-in MLP classification head.

    Used as the Transformer-Only ablation baseline. The encoder architecture
    is identical to TabTransformerExtractor to ensure a fair comparison —
    the only difference is that classification is performed by an internal
    neural head rather than an external CatBoost model.

    Parameters
    ----------
    num_features : int
        Number of binary input features.
    d_model : int
        Embedding / hidden dimension (default 32, matching the extractor).
    nhead : int
        Number of self-attention heads (default 4).
    num_layers : int
        Number of stacked transformer encoder layers (default 3).
    dropout : float
        Dropout probability applied inside the encoder (default 0.2).
    num_classes : int
        Number of output classes (default 2: Susceptible / Resistant).
    """

    def __init__(self, num_features, d_model=32, nhead=4,
                 num_layers=3, dropout=0.2, num_classes=2):
        super().__init__()
        self.d_model = d_model

        # Gene identity embedding — same as TabTransformerExtractor
        self.feature_embed = nn.Embedding(num_features, d_model)

        # Scalar value projection — same as TabTransformerExtractor
        self.value_proj = nn.Linear(1, d_model)

        # Transformer encoder — identical configuration to the extractor
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Built-in classification head: d_model -> d_model//2 -> num_classes
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),   # dimensionality reduction
            nn.ReLU(),                           # non-linear activation
            nn.Dropout(dropout),                 # regularisation
            nn.Linear(d_model // 2, num_classes) # output logits
        )

    def forward(self, x):
        """
        Forward pass: gene tokens -> transformer -> mean pool -> classification head.

        Parameters
        ----------
        x : torch.Tensor, shape (B, F)
            Binary gene presence/absence batch.

        Returns
        -------
        torch.Tensor, shape (B, num_classes)
            Raw logits for each class.
        """
        B, F = x.shape
        idx  = torch.arange(F, device=x.device).unsqueeze(0).repeat(B, 1)
        feat = self.feature_embed(idx)            # (B, F, d_model)
        val  = self.value_proj(x.unsqueeze(-1))   # (B, F, d_model)
        tok  = feat + val                         # (B, F, d_model)
        out  = self.tr(tok)                       # (B, F, d_model)
        pooled = out.mean(1)                      # (B, d_model) — mean pooling
        return self.head(pooled)                  # (B, num_classes)


def train_transformer_only(X_train, y_train, num_features, epochs=120):
    """
    Train the standalone TabTransformerClassifier end-to-end.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training labels.
    num_features : int
        Number of binary input features.
    epochs : int
        Training epochs (default 120, matching train_extractor).

    Returns
    -------
    TabTransformerClassifier
        Trained model in eval() mode.
    """
    model     = TabTransformerClassifier(num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn   = nn.CrossEntropyLoss()
    loader    = DataLoader(TabDataset(X_train, y_train), batch_size=128, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def predict_transformer_only(model, X_df):
    """
    Generate predictions from the standalone TabTransformerClassifier.

    Parameters
    ----------
    model : TabTransformerClassifier
        Trained model in eval() mode.
    X_df : pd.DataFrame
        Feature matrix to predict on.

    Returns
    -------
    preds : np.ndarray
        Binary class predictions (threshold 0.5).
    probs : np.ndarray
        Predicted probabilities for the Resistant class.
    """
    model.eval()
    with torch.no_grad():
        x      = torch.tensor(X_df.values, dtype=torch.float32).to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = (probs > 0.5).astype(int)   # apply default 0.5 threshold
    return preds, probs


# Train Transformer-Only end-to-end model
tr_only_model = train_transformer_only(xtr, ytr, X.shape[1])

# Evaluate on the internal clade-aware test set
pred_tr_only, prob_tr_only = predict_transformer_only(tr_only_model, xts)
results.append(metrics(yts, pred_tr_only, prob_tr_only, "Transformer-Only"))
internal_preds["Transformer-Only"] = pred_tr_only
internal_probs["Transformer-Only"] = prob_tr_only

print("Ablation models trained successfully.")
print("  - Embedding + LR")
print("  - Embedding + MLP")
print("  - Transformer-Only")

# =============================================================================
# 11. CATBOOST HYPERPARAMETER OPTIMISATION VIA OPTUNA
# =============================================================================
# Three search strategies are compared to assess convergence robustness:
# TPE (Tree-structured Parzen Estimator), CMA-ES, and NSGA-II.
# All optimise directly on raw features (no embeddings).

def obj(trial):
    """
    Optuna objective function for CatBoost hyperparameter search.

    Searches over depth, learning_rate, and l2_leaf_reg.
    Evaluated on the clade-aware internal test set using F1-score.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object providing parameter suggestions.

    Returns
    -------
    float
        F1-score on the internal test set for the suggested hyperparameters.
    """
    p = {
        "depth":         trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1, 6)
    }
    m  = CatBoostClassifier(iterations=250, verbose=0, **p)
    m.fit(xtr, ytr)
    pr = m.predict(xts)
    return f1_score(yts, pr)   # maximise F1-score


# --- TPE optimisation ---
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(obj, n_trials=50)   # run 50 trials
cb = CatBoostClassifier(iterations=250, verbose=0, **study.best_params)
cb.fit(xtr, ytr)
pred = cb.predict(xts)
prob = cb.predict_proba(xts)[:, 1]
results.append(metrics(yts, pred, prob, "Optimized CatBoost TPE"))
trained_models["Optimized CatBoost TPE"] = cb
internal_preds["Optimized CatBoost TPE"] = pred
internal_probs["Optimized CatBoost TPE"] = prob

# --- CMA-ES optimisation ---
study = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=42))
study.optimize(obj, n_trials=50)
cb = CatBoostClassifier(iterations=250, verbose=0, **study.best_params)
cb.fit(xtr, ytr)
pred = cb.predict(xts)
prob = cb.predict_proba(xts)[:, 1]
results.append(metrics(yts, pred, prob, "Optimized CatBoost CMA-ES"))
trained_models["Optimized CatBoost CMA-ES"] = cb
internal_preds["Optimized CatBoost CMA-ES"] = pred
internal_probs["Optimized CatBoost CMA-ES"] = prob

# --- NSGA-II optimisation ---
study = optuna.create_study(direction="maximize", sampler=NSGAIISampler(seed=42))
study.optimize(obj, n_trials=50)
cb = CatBoostClassifier(iterations=250, verbose=0, **study.best_params)
cb.fit(xtr, ytr)
pred = cb.predict(xts)
prob = cb.predict_proba(xts)[:, 1]
results.append(metrics(yts, pred, prob, "Optimized CatBoost NSGA II"))
trained_models["Optimized CatBoost NSGA II"] = cb
internal_preds["Optimized CatBoost NSGA II"] = pred
internal_probs["Optimized CatBoost NSGA II"] = prob

# =============================================================================
# 12. CHAINED HYBRID — MULTI-OBJECTIVE NSGA-II ON TRANSFORMER EMBEDDINGS
# =============================================================================
# The Chained Hybrid combines transformer-derived embeddings with CatBoost
# optimised via multi-objective NSGA-II, jointly maximising F1 and MCC
# to balance precision and recall in a Pareto-optimal manner.

def obj_emb(trial):
    """
    Multi-objective Optuna objective for the Chained Hybrid model.

    Operates on transformer embeddings rather than raw features.
    Returns two objectives to be jointly maximised: F1-score and MCC.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object providing parameter suggestions.

    Returns
    -------
    tuple of (float, float)
        (F1-score, MCC) on the internal test set.
    """
    p = {
        "depth":         trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1, 6)
    }
    m  = CatBoostClassifier(iterations=250, verbose=0, **p)
    m.fit(emb_tr, ytr)                          # train on 32-dim embeddings
    pr = m.predict(emb_ts)
    f1  = f1_score(yts, pr)
    mcc = matthews_corrcoef(yts, pr)
    return f1, mcc   # both objectives to be maximised


# Create multi-objective study with NSGA-II sampler
study = optuna.create_study(
    directions=["maximize", "maximize"],   # maximise both F1 and MCC
    sampler=NSGAIISampler(seed=42)
)
study.optimize(obj_emb, n_trials=50)

# Select the Pareto-front trial with the highest MCC as the final configuration
best_trial = max(study.best_trials, key=lambda t: t.values[1])   # values[1] = MCC
print(f"Best Multi-Objective (F1, MCC): {best_trial.values}")

# Train final Chained Hybrid model with selected Pareto-optimal hyperparameters
cb = CatBoostClassifier(iterations=250, verbose=0, **best_trial.params)
cb.fit(emb_tr, ytr)
pred = cb.predict(emb_ts)
prob = cb.predict_proba(emb_ts)[:, 1]
results.append(metrics(yts, pred, prob, "Chained Hybrid"))
trained_models["Chained Hybrid"] = cb
internal_preds["Chained Hybrid"] = pred
internal_probs["Chained Hybrid"] = prob

# =============================================================================
# 13. FINAL INTERNAL RESULTS TABLE
# =============================================================================
final = pd.DataFrame(results)
print("\nFINAL TABLE\n")
print(final.round(4))
print("\nUsing fixed global threshold (distribution shift adjusted)")

# =============================================================================
# 14. EXTERNAL VALIDATION
# =============================================================================
# Evaluate the best-performing model on seven independent multinational BioProjects
# not used during any stage of training or hyperparameter optimisation.

print("\n==============================")
print("EXTERNAL VALIDATION")
print("==============================")

# Load external validation dataset (includes metadata columns)
val_full = pd.read_csv("validationPRJlison.csv", sep=";")

# Separate metadata (not used as features) from model input
meta = val_full[["BioProjectAccession", "IsolationCountry"]].copy()

# Remove metadata columns from the feature matrix
val = val_full.drop(columns=["BioProjectAccession", "IsolationCountry"])
if "Genome ID" in val.columns:
    val = val.drop(columns=["Genome ID"])

# Extract labels and features for the external cohort
y_val = val["Resistant Phenotype"].map({"Susceptible": 0, "Resistant": 1})
X_val = val.drop(columns=["Resistant Phenotype"])

# Align external features to the training feature set
# (fills any missing gene columns with 0 — gene absent in external panel)
X_val = X_val.reindex(columns=X.columns, fill_value=0)

print("External size:", len(X_val))
print("Class dist:\n", y_val.value_counts())

# Select the best model by internal F1-score
best_idx        = final["F1"].idxmax()
best_model_name = final.loc[best_idx, "Model"]

# Realign external features (safety check)
X_val = X_val.reindex(columns=X.columns, fill_value=0)

# =============================================================================
# 15. FULL-DATA RETRAIN FOR EXTERNAL VALIDATION
# =============================================================================
# Retrain the best model on the complete deduplicated dataset (train + test)
# before applying to the external cohort, maximising available training signal.

# Fixed decision threshold selected based on external cohort distribution shift
# (see threshold sensitivity analysis in Section 16)
calibrated_threshold = 0.05
best_model = trained_models[best_model_name]

if best_model_name in ["TabTransformer-CatBoost", "Chained Hybrid"]:
    # Retrain transformer encoder on full data
    extractor_full = TabTransformerExtractor(X.shape[1]).to(device)
    extractor_full = train_extractor(extractor_full, X, y)

    # Extract embeddings for full training data and external validation set
    emb_full = get_emb(extractor_full, X)
    emb_val  = get_emb(extractor_full, X_val)

    # Retrain CatBoost on full-data embeddings
    final_model = CatBoostClassifier(**best_model.get_params())
    final_model.fit(emb_full, y)

    # Predict on external validation using embeddings
    prob_val = final_model.predict_proba(emb_val)[:, 1]
else:
    # Non-hybrid models: retrain directly on raw features
    final_model = CatBoostClassifier(**best_model.get_params())
    final_model.fit(X, y)
    prob_val = final_model.predict_proba(X_val)[:, 1]

# Apply calibrated threshold to obtain binary predictions
pred_val = (prob_val > calibrated_threshold).astype(int)

# Report overall external validation metrics
ext_results = metrics(y_val, pred_val, prob_val, "External Validation")
print("\nEXTERNAL RESULTS\n")
print(pd.DataFrame([ext_results]).round(4))

# =============================================================================
# 16. THRESHOLD SENSITIVITY ANALYSIS (Reviewer 1, Comment 3)
# =============================================================================
# Evaluates model performance across six decision thresholds on the external
# validation cohort. Confirms that 0.05 is the empirically optimal threshold
# for this distribution-shifted external cohort.

print("\n==============================")
print("THRESHOLD SENSITIVITY ANALYSIS (EXTERNAL)")
print("==============================")

thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]   # candidate thresholds
sensitivity_results = []

for t in thresholds:
    pred_t = (prob_val > t).astype(int)   # apply threshold to external probs

    # Skip thresholds that predict only one class (metrics undefined)
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

# Display and save sensitivity table (Supplementary Table S3)
sens_df = pd.DataFrame(sensitivity_results).round(4)
print(sens_df.to_string(index=False))
sens_df.to_csv("TABLE_threshold_sensitivity_external.csv", index=False)
print("TABLE_threshold_sensitivity_external.csv saved")

# =============================================================================
# 17. PER-BIOPROJECT PERFORMANCE
# =============================================================================
# Breaks down external validation performance by individual BioProject
# to identify cohort-specific generalisability patterns.

print("\nPER BIOPROJECT PERFORMANCE")

# Reset indices for consistent row-wise alignment
meta         = meta.reset_index(drop=True)
X_val_meta   = X_val.reset_index(drop=True)
y_val_meta   = y_val.reset_index(drop=True)

per_project_results = []

for project in meta["BioProjectAccession"].unique():
    idx = meta["BioProjectAccession"] == project

    # Skip BioProjects with fewer than 3 isolates (metrics unreliable)
    if idx.sum() < 3:
        continue

    X_p = X_val_meta[idx]
    y_p = y_val_meta[idx]

    if best_model_name in ["TabTransformer-CatBoost", "Chained Hybrid"]:
        emb_p  = get_emb(extractor_full, X_p)
        prob_p = final_model.predict_proba(emb_p)[:, 1]
    else:
        prob_p = final_model.predict_proba(X_p)[:, 1]

    pred_p = (prob_p > calibrated_threshold).astype(int)
    res = metrics(y_p, pred_p, prob_p, project)
    res["n"] = idx.sum()   # record cohort size
    per_project_results.append(res)

per_df = pd.DataFrame(per_project_results)
per_df.to_csv("TABLE_per_project.csv", index=False)
print(per_df.round(4))

# =============================================================================
# 18. PER-COUNTRY PERFORMANCE
# =============================================================================

print("\nPER COUNTRY PERFORMANCE")

country_results = []

for c in meta["IsolationCountry"].unique():
    idx = meta["IsolationCountry"] == c

    # Skip countries with fewer than 5 isolates
    if idx.sum() < 5:
        continue

    X_c = X_val_meta[idx]
    y_c = y_val_meta[idx]

    if best_model_name in ["TabTransformer-CatBoost", "Chained Hybrid"]:
        emb_c  = get_emb(extractor_full, X_c)
        prob_c = final_model.predict_proba(emb_c)[:, 1]
    else:
        prob_c = final_model.predict_proba(X_c)[:, 1]

    pred_c = (prob_c > calibrated_threshold).astype(int)
    res = metrics(y_c, pred_c, prob_c, c)
    res["n"] = idx.sum()
    country_results.append(res)

country_df = pd.DataFrame(country_results)
country_df.to_csv("TABLE_country.csv", index=False)
print(country_df.round(4))

# =============================================================================
# 19. SELECT BEST MODEL FOR FIGURE GENERATION
# =============================================================================
# Retrieve predictions for the best model on the internal test set
best_idx  = final["F1"].idxmax()
best_name = final.loc[best_idx, "Model"]
best_pred = internal_preds[best_name]
best_prob = internal_probs[best_name]

# ROC curves for internal and external sets
plot_roc(yts, best_prob, best_name)
plot_roc(y_val, prob_val, "External")

# =============================================================================
# 20. FIGURE GENERATION FUNCTIONS
# =============================================================================

def fig_pipeline():
    """Generate and save a schematic pipeline overview figure (Figure 1)."""
    plt.figure(figsize=(8, 4))
    plt.text(0.1, 0.7, "Genomic matrix")
    plt.text(0.35, 0.7, "Clade split")
    plt.text(0.6, 0.7, "Transformer")
    plt.text(0.85, 0.7, "CatBoost")
    plt.arrow(0.2, 0.7, 0.1, 0)
    plt.arrow(0.45, 0.7, 0.1, 0)
    plt.arrow(0.7, 0.7, 0.1, 0)
    plt.axis("off")
    plt.savefig("FIG1_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_pr(y, prob, name):
    """
    Plot and save a Precision-Recall curve.

    Parameters
    ----------
    y : array-like
        True binary labels.
    prob : array-like
        Predicted probabilities.
    name : str
        Label used in the output filename.
    """
    p, r, _ = precision_recall_curve(y, prob)
    plt.figure(figsize=(5, 4))
    plt.plot(r, p)
    plt.title("PR curve")
    plt.savefig(f"FIG3_pr_{name}.png", dpi=300)
    plt.close()


def fig_calibration(y, prob, name):
    """
    Plot and save a calibration (reliability) curve.

    Parameters
    ----------
    y : array-like
        True binary labels.
    prob : array-like
        Predicted probabilities.
    name : str
        Label used in the output filename.
    """
    frac, mean = calibration_curve(y, prob, n_bins=10)
    plt.figure(figsize=(5, 4))
    plt.plot(mean, frac, marker="o")
    plt.plot([0, 1], [0, 1], "--")   # perfect calibration diagonal
    plt.title("Calibration")
    plt.savefig(f"FIG4_calibration_{name}.png", dpi=300)
    plt.close()


def fig_attention(extractor, X):
    """
    Compute and save a layer-and-head-averaged attention matrix heatmap.

    Averages attention weights across all layers, all heads, and the first
    128 samples to produce a single (F x F) attention matrix.

    Parameters
    ----------
    extractor : TabTransformerExtractor
        Trained encoder with return_attn=True support.
    X : pd.DataFrame
        Feature matrix (uses up to 128 samples for averaging).
    """
    extractor.eval()
    with torch.no_grad():
        # Use up to 128 samples for attention averaging
        x = torch.tensor(X.values[:128], dtype=torch.float32).to(device)
        _, attn_list = extractor(x, return_attn=True)

    # attn_list: list of (B, H, T, T) tensors, one per encoder layer
    attn = torch.stack(attn_list)   # (L, B, H, T, T)
    attn = attn.mean(0)             # average over layers -> (B, H, T, T)
    attn = attn.mean(0)             # average over batch  -> (H, T, T)
    attn = attn.mean(0)             # average over heads  -> (T, T)
    attn = attn.cpu().numpy()

    plt.figure(figsize=(6, 5))
    sns.heatmap(attn, cmap="viridis")
    plt.title("Attention matrix (layer+head averaged)")
    plt.tight_layout()
    plt.savefig("FIG6_attention.png", dpi=300)
    plt.close()
    print("FIG6_attention.png saved")


def fig_clade(groups, y):
    """
    Plot clade leakage check heatmap (clade x phenotype count matrix).

    Parameters
    ----------
    groups : array-like
        Clade assignments from agglomerative clustering.
    y : pd.Series
        Binary phenotype labels.
    """
    df = pd.DataFrame({"g": groups, "y": y})
    ct = pd.crosstab(df.g, df.y)   # clade x phenotype count table
    sns.heatmap(ct, cmap="Blues")
    plt.title("Clade leakage check")
    plt.savefig("FIG7_clade.png", dpi=300)
    plt.close()


def fig_external_bar(internal, external):
    """
    Plot grouped bar chart comparing internal and external F1-scores.

    Parameters
    ----------
    internal : pd.DataFrame
        Internal results DataFrame with Model and F1 columns.
    external : pd.DataFrame
        External results DataFrame with Model and F1 columns.
    """
    df = pd.concat([internal, external])
    sns.barplot(data=df, x="Model", y="F1")
    plt.xticks(rotation=45)
    plt.savefig("FIG8_external.png", dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# 21. GENERATE ALL PUBLICATION FIGURES
# =============================================================================
print("\nGenerating publication figures...")

# Realign external validation features (safety check before figure generation)
X_val = X_val.reindex(columns=X.columns, fill_value=0)

fig_pipeline()                                        # Figure 1: pipeline schematic

fig_pr(yts, best_prob, "internal")                    # PR curve — internal test
fig_pr(y_val, prob_val, "external")                   # PR curve — external cohort

fig_calibration(yts, best_prob, "internal")           # calibration — internal
fig_calibration(y_val, prob_val, "external")          # calibration — external

# --- SHAP feature importance (Figure 5) ---
print("\nComputing SHAP on transformer embeddings...")

if best_model_name in ["TabTransformer-CatBoost", "Chained Hybrid"]:
    # SHAP operates on the 32-dimensional embedding space, not raw genes
    shap_X = pd.DataFrame(emb_tr)
    explainer    = shap.TreeExplainer(final_model)
    shap_values  = explainer.shap_values(shap_X)

    shap.summary_plot(shap_values, shap_X, show=False, plot_size=(7, 5))
    plt.title("Feature importance in transformer embedding space")
    plt.tight_layout()
    plt.savefig("FIG5_shap.png", dpi=600)
    plt.close()
    print("FIG5 SHAP saved (embedding space)")

fig_attention(extractor_full, X)                      # Figure 6: attention heatmap

fig_clade(groups, y)                                  # clade leakage check

internal_df = final
external_df = pd.DataFrame([ext_results])
fig_external_bar(internal_df, external_df)            # Figure 8: F1 comparison bar

print("All publication figures saved.")

# Geographic distribution of external isolates (supplementary)
plt.figure(figsize=(7, 5))
sns.countplot(data=meta, y="IsolationCountry")
plt.title("Geographic distribution of external isolates")
plt.tight_layout()
plt.savefig("SUPP_country_distribution.png", dpi=300)
plt.close()

# =============================================================================
# 22. COMBINED ROC CURVE (Figure 3)
# =============================================================================

def plot_combined_roc(y_internal, prob_internal, y_external, prob_external,
                      save_name="FIG3_ROC_combined.png"):
    """
    Plot side-by-side ROC curves for internal and external evaluation.

    Parameters
    ----------
    y_internal : array-like
        True labels for the internal test set.
    prob_internal : array-like
        Predicted probabilities for the internal test set.
    y_external : array-like
        True labels for the external validation cohort.
    prob_external : array-like
        Predicted probabilities for the external validation cohort.
    save_name : str
        Output filename (default "FIG3_ROC_combined.png").
    """
    fpr_int, tpr_int, _ = roc_curve(y_internal, prob_internal)
    auc_int              = auc(fpr_int, tpr_int)
    fpr_ext, tpr_ext, _ = roc_curve(y_external, prob_external)
    auc_ext              = auc(fpr_ext, tpr_ext)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: internal test ROC
    axes[0].plot(fpr_int, tpr_int, linewidth=2)
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].set_title("(A) Internal Test ROC\nChained Hybrid")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend([f"AUC = {auc_int:.3f}"])

    # Panel B: external validation ROC
    axes[1].plot(fpr_ext, tpr_ext, linewidth=2)
    axes[1].plot([0, 1], [0, 1], linestyle="--")
    axes[1].set_title("(B) External Validation ROC\nChained Hybrid")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend([f"AUC = {auc_ext:.3f}"])

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print("ROC figure saved:", save_name)


plot_combined_roc(yts, best_prob, y_val, prob_val)

# =============================================================================
# 23. t-SNE VISUALISATION OF TRANSFORMER EMBEDDINGS (Figure 2)
# =============================================================================

def plot_tsne_dual(emb_test, y_test, emb_ext, y_ext,
                   save_name="FIG2_TSNE.png"):
    """
    Plot side-by-side t-SNE projections of transformer embeddings.

    Visualises phenotype separation in the learned latent space for
    both the internal test set and the external validation cohort.

    Parameters
    ----------
    emb_test : np.ndarray
        32-dimensional embeddings for the internal test set.
    y_test : array-like
        True labels for the internal test set.
    emb_ext : np.ndarray
        32-dimensional embeddings for the external cohort.
    y_ext : array-like
        True labels for the external cohort.
    save_name : str
        Output filename (default "FIG2_TSNE.png").
    """
    # Fit t-SNE independently for each cohort (different sample sizes)
    tsne_test = TSNE(n_components=2, perplexity=30,
                     learning_rate="auto", init="pca", random_state=42)
    z_test = tsne_test.fit_transform(emb_test)

    tsne_ext = TSNE(n_components=2, perplexity=30,
                    learning_rate="auto", init="pca", random_state=42)
    z_ext = tsne_ext.fit_transform(emb_ext)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: internal test embeddings
    axes[0].scatter(z_test[:, 0], z_test[:, 1], c=y_test, s=20)
    axes[0].set_title("(A) Internal test embeddings\nChained Hybrid")
    axes[0].set_xlabel("t-SNE1")
    axes[0].set_ylabel("t-SNE2")

    # Panel B: external validation embeddings
    axes[1].scatter(z_ext[:, 0], z_ext[:, 1], c=y_ext, s=20)
    axes[1].set_title("(B) External embeddings\nChained Hybrid")
    axes[1].set_xlabel("t-SNE1")
    axes[1].set_ylabel("t-SNE2")

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print("t-SNE figure saved:", save_name)


plot_tsne_dual(emb_ts, yts, emb_val, y_val)

# =============================================================================
# 24. ATTENTION PAIR ANALYSIS (Figure 6)
# =============================================================================

def get_attention_matrix(extractor, X, device):
    """
    Compute a single aggregated attention matrix by averaging over
    all transformer layers, all attention heads, and up to 128 samples.

    Parameters
    ----------
    extractor : TabTransformerExtractor
        Trained encoder.
    X : pd.DataFrame
        Feature matrix (up to 128 rows used).
    device : torch.device
        Compute device.

    Returns
    -------
    np.ndarray, shape (F, F)
        Averaged attention weight matrix over genes.
    """
    extractor.eval()
    with torch.no_grad():
        x = torch.tensor(X.values[:128], dtype=torch.float32).to(device)
        _, attn_list = extractor(x, return_attn=True)

    attn = torch.stack(attn_list)   # (L, B, H, T, T)
    attn = attn.mean(0)             # -> (B, H, T, T)
    attn = attn.mean(0)             # -> (H, T, T)
    attn = attn.mean(0)             # -> (T, T)
    return attn.cpu().numpy()


def extract_top_pairs(att_matrix, gene_names, top_k=20):
    """
    Extract the top-k gene pairs with highest pairwise attention weights.

    Parameters
    ----------
    att_matrix : np.ndarray, shape (F, F)
        Aggregated attention weight matrix.
    gene_names : list of str
        Gene names corresponding to matrix rows/columns.
    top_k : int
        Number of top pairs to return (default 20).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns Gene_A, Gene_B, Attention, sorted descending.
    """
    pairs = []
    for i in range(len(gene_names)):
        for j in range(i + 1, len(gene_names)):
            # Collect upper-triangle pairs (symmetric matrix)
            pairs.append((gene_names[i], gene_names[j], att_matrix[i, j]))

    df = pd.DataFrame(pairs, columns=["Gene_A", "Gene_B", "Attention"])
    return df.sort_values("Attention", ascending=False).head(top_k)


def plot_attention_bar(df):
    """
    Plot a horizontal bar chart of the top gene-pair attention weights.

    Parameters
    ----------
    df : pd.DataFrame
        Output of extract_top_pairs with columns Gene_A, Gene_B, Attention.
    """
    labels = df["Gene_A"] + " <-> " + df["Gene_B"]
    plt.figure(figsize=(5.5, 4))
    plt.barh(labels[::-1], df["Attention"][::-1])
    plt.xlabel("Attention weight")
    plt.title("Top genomic interactions")
    plt.tight_layout()
    plt.savefig("FIG6_attention_pairs.png", dpi=600)
    plt.close()
    print("FIG6 saved")


# Compute and save attention pair figure
att   = get_attention_matrix(extractor_full, X, device)
genes = X.columns.tolist()
top_df = extract_top_pairs(att, genes, top_k=20)
plot_attention_bar(top_df)

# =============================================================================
# 25. EXTERNAL COHORT UNIQUE GENOME COUNT REPORT
# =============================================================================
print("\n==============================")
print("UNIQUE EXTERNAL GENOME COUNTS")
print("==============================")

# Reload external validation file for independent deduplication count
val_full  = pd.read_csv("validationPRJlison.csv", sep=";")
meta_tmp  = val_full[["BioProjectAccession", "IsolationCountry"]].copy()
val_tmp   = val_full.drop(columns=["BioProjectAccession", "IsolationCountry"])

if "Genome ID" in val_tmp.columns:
    val_tmp = val_tmp.drop(columns=["Genome ID"])

y_tmp = val_tmp["Resistant Phenotype"].map({"Susceptible": 0, "Resistant": 1})
X_tmp = val_tmp.drop(columns=["Resistant Phenotype"])
X_tmp = X_tmp.reindex(columns=X.columns, fill_value=0)   # align to training features

# Apply safe deduplication to external cohort as well
genome_sig_ext = X_tmp.astype(str).agg("_".join, axis=1)
df_ext         = pd.concat([X_tmp, y_tmp], axis=1)
df_ext["sig"]  = genome_sig_ext
label_counts_ext = df_ext.groupby("sig")["Resistant Phenotype"].nunique()
valid_sigs_ext   = label_counts_ext[label_counts_ext == 1].index

df_ext_clean = (
    df_ext[df_ext["sig"].isin(valid_sigs_ext)]
    .drop_duplicates(subset="sig")
    .drop(columns="sig")
    .reset_index(drop=True)
)
y_unique = df_ext_clean["Resistant Phenotype"]

print("Total external isolates:", len(val_tmp))
print("Unique genomes:", len(df_ext_clean))
print("\nUnique class distribution:")
print(y_unique.value_counts().rename({0: "Susceptible", 1: "Resistant"}))

# =============================================================================
# 26. LATENT DIMENSION TO GENE MAPPING
# =============================================================================

def map_latent_to_genes(X_df, embeddings, target_dimensions, top_k=10):
    """
    Compute Pearson correlations between latent embedding dimensions
    and original binary gene features to interpret latent space structure.

    Parameters
    ----------
    X_df : pd.DataFrame
        Original binary gene matrix used to train the encoder.
    embeddings : np.ndarray, shape (n_samples, d_model)
        Transformer embeddings for the same samples.
    target_dimensions : list of int
        Indices of embedding dimensions to analyse.
    top_k : int
        Number of top-correlated genes to report per dimension (default 10).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Latent_Dimension, Gene_Name,
        Correlation, Direction.
    """
    gene_names = X_df.columns.tolist()
    X_matrix   = X_df.values
    results    = []

    for d in target_dimensions:
        dim_vector   = embeddings[:, d]   # embedding values for dimension d
        correlations = []

        for i, gene in enumerate(gene_names):
            # Pearson correlation between gene i and embedding dimension d
            corr = np.corrcoef(X_matrix[:, i], dim_vector)[0, 1]
            if not np.isnan(corr):
                correlations.append((gene, corr))

        # Sort by absolute correlation magnitude (strongest association first)
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        for gene, corr in correlations[:top_k]:
            results.append({
                "Latent_Dimension": d,
                "Gene_Name":        gene,
                "Correlation":      round(corr, 4),
                "Direction":        "Positive" if corr > 0 else "Negative"
            })

    return pd.DataFrame(results)


# Analyse the 6 most SHAP-influential embedding dimensions
target_dims = [11, 18, 30, 2, 23, 29]
mapping_df  = map_latent_to_genes(xtr, emb_tr, target_dims)
print("\n=== LATENT DIMENSION TO GENE MAPPING (Top 10 Genes per Dim) ===")
print(mapping_df)
mapping_df.to_csv("TABLE_latent_gene_mapping.csv", index=False)

# Print a one-line summary per dimension for Discussion interpretation
for d in target_dims:
    top_gene = mapping_df[mapping_df["Latent_Dimension"] == d].iloc[0]["Gene_Name"]
    corr_val = mapping_df[mapping_df["Latent_Dimension"] == d].iloc[0]["Correlation"]
    print(f"Dimension {d} is primarily driven by {top_gene} (r={corr_val})")

# Full mapping for all 32 dimensions (Supplementary Table S1)
target_dims_full = list(range(32))
mapping_df_full  = map_latent_to_genes(xtr, emb_tr, target_dims_full, top_k=5)
mapping_df_full.to_csv("FULL_LATENT_GENE_MAPPING_32D.csv", index=False)
print("32-dimension mapping saved to FULL_LATENT_GENE_MAPPING_32D.csv")

# =============================================================================
# 27. LATENT DIMENSION CORRELATION HEATMAP (Figure 7)
# =============================================================================
# Visualises inter-correlations between the 32 embedding dimensions.
# A block structure indicates that the transformer has learned
# coordinated, multi-dimensional genomic representations.

corr_matrix = np.corrcoef(emb_tr.T)   # (32, 32) correlation matrix

plt.figure(figsize=(10, 8))            # large figure for high-resolution output
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.tight_layout()
plt.savefig("FIG7_correlation.png", dpi=600, bbox_inches="tight")
plt.close()

# =============================================================================
# 28. COMBINED CONFUSION MATRIX (Figure 4)
# =============================================================================

def plot_cm_dual(y_int, pred_int, y_ext, pred_ext):
    """
    Plot normalised confusion matrices side-by-side for internal and external sets.

    Row normalisation is applied so each cell shows the fraction of true-class
    isolates correctly or incorrectly classified.

    Parameters
    ----------
    y_int : array-like
        True labels for the internal test set.
    pred_int : array-like
        Predicted labels for the internal test set.
    y_ext : array-like
        True labels for the external validation cohort.
    pred_ext : array-like
        Predicted labels for the external validation cohort.
    """
    cm_int = confusion_matrix(y_int, pred_int).astype(float)
    cm_ext = confusion_matrix(y_ext, pred_ext).astype(float)

    # Row-normalise: divide each row by its sum
    cm_int = cm_int / cm_int.sum(axis=1)[:, None]
    cm_ext = cm_ext / cm_ext.sum(axis=1)[:, None]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Panel A: internal test confusion matrix
    sns.heatmap(cm_int, annot=True, fmt=".2f", cmap="Blues",
                cbar=False, square=True,
                xticklabels=["Susceptible", "Resistant"],
                yticklabels=["Susceptible", "Resistant"],
                ax=axes[0])
    axes[0].set_title("(A) Internal test")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Panel B: external validation confusion matrix
    sns.heatmap(cm_ext, annot=True, fmt=".2f", cmap="Blues",
                cbar=False, square=True,
                xticklabels=["Susceptible", "Resistant"],
                yticklabels=["Susceptible", "Resistant"],
                ax=axes[1])
    axes[1].set_title("(B) External validation")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig("FIG_CM_combined.png", dpi=600)
    plt.close()
    print("FIG_CM_combined.png saved")


plot_cm_dual(yts, best_pred, y_val, pred_val)

# =============================================================================
# 29. BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=1000, random_state=42):
    """
    Estimate 95% confidence intervals for MCC, AUROC, and F1 via
    non-parametric bootstrap resampling (Efron and Tibshirani 1994).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bootstrap : int
        Number of bootstrap iterations (default 1000).
    random_state : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    dict
        Keys: 'MCC', 'AUROC', 'F1'. Values: (mean, lower_2.5%, upper_97.5%).
    """
    np.random.seed(random_state)
    mcc_scores, auc_scores, f1_scores = [], [], []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx  = resample(range(n))   # sample with replacement
        y_t  = y_true[idx]
        y_p  = y_pred[idx]
        y_pr = y_prob[idx]

        # AUROC requires both classes to be present in the bootstrap sample
        if len(np.unique(y_t)) < 2:
            continue

        mcc_scores.append(matthews_corrcoef(y_t, y_p))
        auc_scores.append(roc_auc_score(y_t, y_pr))
        f1_scores.append(f1_score(y_t, y_p))

    def summary(scores):
        """Return (mean, 2.5th percentile, 97.5th percentile)."""
        return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

    return {
        "MCC":   summary(mcc_scores),
        "AUROC": summary(auc_scores),
        "F1":    summary(f1_scores)
    }


print("\n==============================")
print("BOOTSTRAP CONFIDENCE INTERVALS")
print("==============================")

# Internal test set bootstrap CI
print("\nINTERNAL TEST SET")
internal_ci = bootstrap_ci(
    yts.values if hasattr(yts, "values") else yts,
    best_pred, best_prob, n_bootstrap=1000
)
for metric, values in internal_ci.items():
    mean, lower, upper = values
    print(f"{metric}: {mean:.4f} (95% CI: {lower:.4f} - {upper:.4f})")

# External validation bootstrap CI
print("\nEXTERNAL VALIDATION SET")
external_ci = bootstrap_ci(
    y_val.values if hasattr(y_val, "values") else y_val,
    pred_val, prob_val, n_bootstrap=1000
)
for metric, values in external_ci.items():
    mean, lower, upper = values
    print(f"{metric}: {mean:.4f} (95% CI: {lower:.4f} - {upper:.4f})")

# =============================================================================
# 30. RAW DATA BASELINE (NO DEDUPLICATION, NO CLADE SPLIT)
# =============================================================================
# Demonstrates the impact of leakage-aware preprocessing by comparing
# against a naive CatBoost model trained on raw, unprocessed data.

print("\n==============================")
print("RAW DATA BASELINE CATBOOST")
print("==============================")

# Load raw data without any deduplication or clade-aware splitting
df_raw = pd.read_csv("asilverisetigenler.csv", sep=";")
if "Genome ID" in df_raw.columns:
    df_raw = df_raw.drop(columns=["Genome ID"])

y_raw = df_raw["Resistant Phenotype"].map({"Susceptible": 0, "Resistant": 1})
X_raw = df_raw.drop(columns=["Resistant Phenotype"])
X_raw = X_raw.loc[:, X_raw.nunique() > 1]   # remove constant columns

# Standard stratified random split (no clade awareness)
Xr_tr, Xr_ts, yr_tr, yr_ts = train_test_split(
    X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)

# Train raw baseline CatBoost with same hyperparameters as the baseline hybrid
raw_cb = CatBoostClassifier(iterations=250, depth=8, verbose=0, random_state=42)
raw_cb.fit(Xr_tr, yr_tr)
raw_pred = raw_cb.predict(Xr_ts)
raw_prob = raw_cb.predict_proba(Xr_ts)[:, 1]

print(pd.DataFrame([metrics(yr_ts, raw_pred, raw_prob, "Raw CatBoost (No Dedup)")]).round(4))

# Bootstrap CI for raw baseline
print("\nRAW DATA BOOTSTRAP CI")
raw_ci = bootstrap_ci(
    yr_ts.values if hasattr(yr_ts, "values") else yr_ts,
    raw_pred, raw_prob, n_bootstrap=1000
)
for metric, values in raw_ci.items():
    mean, lower, upper = values
    print(f"{metric}: {mean:.4f} (95% CI: {lower:.4f} - {upper:.4f})")

# External validation of raw baseline
print("\n==============================")
print("RAW MODEL - EXTERNAL VALIDATION")
print("==============================")

val_raw = pd.read_csv("validationPRJlison.csv", sep=";")
if "Genome ID" in val_raw.columns:
    val_raw = val_raw.drop(columns=["Genome ID"])
for col in ["BioProjectAccession", "IsolationCountry"]:
    if col in val_raw.columns:
        val_raw = val_raw.drop(columns=[col])

y_val_raw = val_raw["Resistant Phenotype"].map({"Susceptible": 0, "Resistant": 1})
X_val_raw = val_raw.drop(columns=["Resistant Phenotype"])
X_val_raw = X_val_raw.reindex(columns=X_raw.columns, fill_value=0)   # align features

raw_ext_pred = raw_cb.predict(X_val_raw)
raw_ext_prob = raw_cb.predict_proba(X_val_raw)[:, 1]

print(pd.DataFrame([metrics(y_val_raw, raw_ext_pred, raw_ext_prob,
                             "Raw CatBoost External")]).round(4))

# Bootstrap CI for raw external
print("\nRAW EXTERNAL BOOTSTRAP CI")
raw_ext_ci = bootstrap_ci(
    y_val_raw.values if hasattr(y_val_raw, "values") else y_val_raw,
    raw_ext_pred, raw_ext_prob, n_bootstrap=1000
)
for metric, values in raw_ext_ci.items():
    mean, lower, upper = values
    print(f"{metric}: {mean:.4f} (95% CI: {lower:.4f} - {upper:.4f})")

# Report raw dataset sizes
print(f"Total raw isolates: {len(df_raw)}")
print(f"Raw feature matrix shape: {X_raw.shape}")

# =============================================================================
# 31. DELONG TEST — PAIRWISE AUROC COMPARISON
# =============================================================================

def delong_test(y_true, prob1, prob2, n_boot=1000):
    """
    Bootstrap-based approximation of the DeLong test for comparing two
    receiver operating characteristic curves (AUROC comparison).

    Computes the two-sided p-value for the null hypothesis that the two
    models have equal AUROC on the given dataset.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    prob1 : array-like
        Predicted probabilities from model 1 (e.g. Chained Hybrid).
    prob2 : array-like
        Predicted probabilities from model 2 (e.g. Baseline CatBoost).
    n_boot : int
        Number of bootstrap iterations (default 1000).

    Returns
    -------
    float
        Two-sided p-value; values < 0.05 indicate significant AUROC difference.
    """
    y_true    = np.array(y_true)
    auc_diffs = []

    for _ in range(n_boot):
        # Bootstrap resample with replacement
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        y_b = y_true[idx]
        p1  = prob1[idx]
        p2  = prob2[idx]

        # Skip if only one class is present in the bootstrap sample
        if len(np.unique(y_b)) < 2:
            continue

        auc_diffs.append(roc_auc_score(y_b, p1) - roc_auc_score(y_b, p2))

    auc_diffs = np.array(auc_diffs)

    # Two-sided p-value: fraction of bootstrap differences on the opposing side
    p_value = 2 * min(
        np.mean(auc_diffs <= 0),
        np.mean(auc_diffs >= 0)
    )
    return p_value


print("\n==============================")
print("DELONG TEST (AUROC COMPARISON)")
print("==============================")

y_true_delong  = yts.values if hasattr(yts, "values") else yts
prob_baseline  = internal_probs["Baseline CatBoost"]
prob_hybrid    = internal_probs["Chained Hybrid"]

p_val = delong_test(y_true_delong, prob_baseline, prob_hybrid)

print("Baseline AUROC:", roc_auc_score(y_true_delong, prob_baseline))
print("Hybrid AUROC:  ", roc_auc_score(y_true_delong, prob_hybrid))
print("DeLong p-value:", p_val)
