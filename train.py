"""
MoA Prediction — Transductive whitened PCA + rich features + LogReg ensemble

Key improvements over previous approach:
  - Transductive PCA: fit on combined train+test (better generalization)
  - Whitened PCA: unit-variance components (better LR conditioning)
  - LBFGS multiprocessing: true 48-core parallelism (much faster)
  - Richer features: PCA(120 gene + 50 cell) + interactions(60) + cross-PCA(25)
    + row stats(14) + condition(6) = 275-dim
  - 7 LR models blended (5 base C values + 2 Nystroem)
  - Per-target adaptive Bayesian calibration
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kurtosis as kurtosis_fn
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
t0 = time.time()

# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
train_features = pd.read_csv("data/train_features.csv")
train_targets   = pd.read_csv("data/train_targets.csv")
test_features   = pd.read_csv("data/test_features.csv")
target_cols = [c for c in train_targets.columns if c != "sig_id"]

is_ctrl_tr = train_features["cp_type"] == "ctl_vehicle"
is_ctrl_te = test_features["cp_type"]  == "ctl_vehicle"
Xtr = train_features[~is_ctrl_tr].copy().reset_index(drop=True)
Ytr = train_targets[~is_ctrl_tr][target_cols].values.astype(np.float32)
Xte = test_features[~is_ctrl_te].copy().reset_index(drop=True)

gene_cols = [c for c in train_features.columns if c.startswith("g-")]
cell_cols  = [c for c in train_features.columns if c.startswith("c-")]

# ── transductive whitened PCA (fit on combined train+test) ────────────────────
print("Building transductive whitened PCA features...")
all_feats = pd.concat([train_features, test_features], ignore_index=True)

pca_g  = PCA(n_components=120, whiten=True, random_state=42)
pca_c  = PCA(n_components=50,  whiten=True, random_state=42)
pca_gc = PCA(n_components=50,  whiten=True, random_state=42)
g_all  = pca_g.fit_transform(all_feats[gene_cols].values)              # (N, 120)
c_all  = pca_c.fit_transform(all_feats[cell_cols].values)              # (N,  50)
gc_all = pca_gc.fit_transform(all_feats[gene_cols + cell_cols].values) # (N,  50)

n_train = len(train_features)
# Split back
g_tr  = g_all[:n_train][~is_ctrl_tr.values]   # treatment train
c_tr  = c_all[:n_train][~is_ctrl_tr.values]
gc_tr = gc_all[:n_train][~is_ctrl_tr.values]
g_te  = g_all[n_train:][~is_ctrl_te.values]   # treatment test
c_te  = c_all[n_train:][~is_ctrl_te.values]
gc_te = gc_all[n_train:][~is_ctrl_te.values]

# ── feature engineering ────────────────────────────────────────────────────────
def make_features(df, g_pca, c_pca, gc_pca):
    """Build features from pre-computed whitened PCA arrays."""
    t    = df["cp_time"].values.astype(float)
    dose = (df["cp_dose"] == "D2").astype(float).values
    tnorm = (t - 24) / 48
    t_oh  = np.column_stack([(t == v).astype(float) for v in [24, 48, 72]])
    cond  = np.column_stack([tnorm, dose, tnorm * dose, t_oh])  # (n, 6)

    # Gene-PCA × condition interactions (top-20 gene PCs = 60 features)
    ints = np.hstack([
        g_pca[:, :20] * tnorm[:, None],
        g_pca[:, :20] * dose[:, None],
        g_pca[:, :20] * (tnorm * dose)[:, None],
    ])

    # Cross-PCA: top-5 gene × top-5 cell (25 features)
    cross = (g_pca[:, :5, None] * c_pca[:, None, :5]).reshape(len(df), 25)

    # Per-row gene/cell statistics (14 features)
    raw_g = df[gene_cols].values
    raw_c = df[cell_cols].values
    g_kurt = kurtosis_fn(raw_g, axis=1).reshape(-1, 1)
    c_kurt = kurtosis_fn(raw_c, axis=1).reshape(-1, 1)
    g_pct  = np.percentile(raw_g, [10, 25, 50, 75, 90], axis=1).T
    c_pct  = np.percentile(raw_c, [10, 25, 50, 75, 90], axis=1).T
    g_iqr  = (g_pct[:, 3] - g_pct[:, 1]).reshape(-1, 1)
    c_iqr  = (c_pct[:, 3] - c_pct[:, 1]).reshape(-1, 1)

    # Total: 120 + 50 + 50 + 60 + 25 + 14 + 6 = 325
    return np.hstack([g_pca, c_pca, gc_pca, ints, cross,
                      g_kurt, c_kurt, g_pct, c_pct, g_iqr, c_iqr,
                      cond]).astype(np.float32)

print("Building features...")
Xtr_f = make_features(Xtr, g_tr, c_tr, gc_tr)
Xte_f = make_features(Xte, g_te, c_te, gc_te)
print(f"  base dim={Xtr_f.shape[1]}  ({time.time()-t0:.1f}s)")

fsc   = StandardScaler().fit(Xtr_f)
Xtr_s = fsc.transform(Xtr_f)
Xte_s = fsc.transform(Xte_f)

# Nystroem kernel approximation (gamma=0.05: empirically provides best diversity)
print("Fitting Nystroem kernel...")
nys = Nystroem(kernel='rbf', gamma=0.05, n_components=300, random_state=42).fit(Xtr_s)
Xtr_nys = nys.transform(Xtr_s).astype(np.float32)
Xte_nys = nys.transform(Xte_s).astype(np.float32)
nys_fsc = StandardScaler().fit(Xtr_nys)
Xtr_nys = nys_fsc.transform(Xtr_nys)
Xte_nys = nys_fsc.transform(Xte_nys)
print(f"  Nystroem dim={Xtr_nys.shape[1]}  ({time.time()-t0:.1f}s)")

# ── parallel per-target fitting (LBFGS with multiprocessing) ──────────────────
def fit_lr(X_tr, y_col, C):
    y = y_col.astype(int)
    if len(np.unique(y)) < 2:
        clf = DummyClassifier(strategy="most_frequent")
    else:
        clf = LogisticRegression(C=C, max_iter=500, solver='lbfgs')
    clf.fit(X_tr, y)
    return clf

def predict_clf(clf, X_te):
    classes = list(clf.classes_) if hasattr(clf, 'classes_') else [0, 1]
    if 1 not in classes:
        return np.zeros(len(X_te))
    p = clf.predict_proba(X_te)
    return p[:, classes.index(1)]

def sweep(X_tr, X_te, C, label):
    t1 = time.time()
    ests = Parallel(n_jobs=-1)(
        delayed(fit_lr)(X_tr, Ytr[:, j], C)
        for j in range(len(target_cols))
    )
    pred = np.column_stack([predict_clf(e, X_te) for e in ests]).astype(np.float32)
    print(f"  {label}: {time.time()-t1:.1f}s")
    return pred

# Warm up worker pool
print("Warming up worker pool...")
Parallel(n_jobs=-1)([delayed(lambda: 1)() for _ in range(48)])

# Base LogReg (5 C values for diversity)
print(f"Base LogReg sweeps... ({time.time()-t0:.1f}s elapsed)")
base_preds = []
for C in [0.02, 0.05, 0.10, 0.20, 0.50]:
    base_preds.append(sweep(Xtr_s, Xte_s, C, f"LR base C={C}"))

# Nystroem kernel LogReg
print(f"Nystroem LogReg... ({time.time()-t0:.1f}s elapsed)")
nys_preds = []
for C in [0.1, 0.5]:
    nys_preds.append(sweep(Xtr_nys, Xte_nys, C, f"LR Nystroem C={C}"))

# Polynomial kernel Nystroem (captures 2nd-order feature interactions)
print("Fitting Poly Nystroem...")
nys_poly = Nystroem(kernel='poly', degree=2, coef0=1,
                    gamma=1.0/Xtr_s.shape[1],
                    n_components=300, random_state=42).fit(Xtr_s)
Xtr_poly = nys_poly.transform(Xtr_s).astype(np.float32)
Xte_poly = nys_poly.transform(Xte_s).astype(np.float32)
poly_fsc = StandardScaler().fit(Xtr_poly)
Xtr_poly = poly_fsc.transform(Xtr_poly)
Xte_poly = poly_fsc.transform(Xte_poly)
print(f"  Poly Nystroem dim={Xtr_poly.shape[1]}  ({time.time()-t0:.1f}s)")

print(f"Poly Nystroem LogReg... ({time.time()-t0:.1f}s elapsed)")
poly_preds = []
for C in [0.1, 0.5]:
    poly_preds.append(sweep(Xtr_poly, Xte_poly, C, f"LR Poly C={C}"))

# MLP on compact features (50-dim gc_pca + 6-dim condition = 56-dim)
# Non-linear patterns that LR ensemble misses; ~100s at 56-dim (well within budget)
print(f"MLP ensemble... ({time.time()-t0:.1f}s elapsed)")

def make_cond(df):
    t = df["cp_time"].values.astype(float)
    dose = (df["cp_dose"] == "D2").astype(float).values
    tnorm = (t - 24) / 48
    t_oh = np.column_stack([(t == v).astype(float) for v in [24, 48, 72]])
    return np.column_stack([tnorm, dose, tnorm * dose, t_oh]).astype(np.float32)

Xtr_mlp = np.hstack([gc_tr, make_cond(Xtr)])
Xte_mlp = np.hstack([gc_te, make_cond(Xte)])
mlp_fsc = StandardScaler().fit(Xtr_mlp)
Xtr_mlp_s = mlp_fsc.transform(Xtr_mlp)
Xte_mlp_s = mlp_fsc.transform(Xte_mlp)

def fit_mlp(X_tr, y_col, hidden, alpha):
    y = y_col.astype(int)
    if len(np.unique(y)) < 2:
        return DummyClassifier(strategy="most_frequent").fit(X_tr, y)
    return MLPClassifier(
        hidden_layer_sizes=hidden, solver='lbfgs', max_iter=500,
        alpha=alpha, random_state=42
    ).fit(X_tr, y)

def sweep_mlp(X_tr, X_te, label, hidden=(64,), alpha=0.001):
    t1 = time.time()
    ests = Parallel(n_jobs=-1)(
        delayed(fit_mlp)(X_tr, Ytr[:, j], hidden, alpha)
        for j in range(len(target_cols))
    )
    pred = np.column_stack([predict_clf(e, X_te) for e in ests]).astype(np.float32)
    print(f"  {label}: {time.time()-t1:.1f}s")
    return pred

mlp_preds = [
    sweep_mlp(Xtr_mlp_s, Xte_mlp_s, "MLP(64,) gc56"),           # compact 56-dim
    sweep_mlp(Xtr_s, Xte_s, "MLP(64,) full325"),                  # full 325-dim
    sweep_mlp(Xtr_mlp_s, Xte_mlp_s, "MLP(128,) gc56", (128,)),   # larger hidden
]

# LR on raw features (captures individual gene patterns beyond PCA top-120)
# 772 gene + 100 cell + 6 cond = 878-dim; different signal from PCA-based models
print(f"Raw feature LR... ({time.time()-t0:.1f}s elapsed)")
Xtr_raw = np.hstack([Xtr[gene_cols + cell_cols].values, make_cond(Xtr)]).astype(np.float32)
Xte_raw = np.hstack([Xte[gene_cols + cell_cols].values, make_cond(Xte)]).astype(np.float32)
raw_fsc = StandardScaler().fit(Xtr_raw)
Xtr_raw_s = raw_fsc.transform(Xtr_raw)
Xte_raw_s = raw_fsc.transform(Xte_raw)

raw_preds = []
for C in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
    raw_preds.append(sweep(Xtr_raw_s, Xte_raw_s, C, f"LR raw C={C}"))

# Gene-only raw features (772-dim): more gene-expression-specific signal
Xtr_gene = np.hstack([Xtr[gene_cols].values, make_cond(Xtr)]).astype(np.float32)
Xte_gene = np.hstack([Xte[gene_cols].values, make_cond(Xte)]).astype(np.float32)
gene_fsc = StandardScaler().fit(Xtr_gene)
Xtr_gene_s = gene_fsc.transform(Xtr_gene)
Xte_gene_s = gene_fsc.transform(Xte_gene)

for C in [0.02, 0.05]:
    raw_preds.append(sweep(Xtr_gene_s, Xte_gene_s, C, f"LR gene C={C}"))

# Cell-only raw features (100-dim): cell viability signal separate from gene expression
Xtr_cell = np.hstack([Xtr[cell_cols].values, make_cond(Xtr)]).astype(np.float32)
Xte_cell = np.hstack([Xte[cell_cols].values, make_cond(Xte)]).astype(np.float32)
cell_fsc = StandardScaler().fit(Xtr_cell)
Xtr_cell_s = cell_fsc.transform(Xtr_cell)
Xte_cell_s = cell_fsc.transform(Xte_cell)

for C in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
    raw_preds.append(sweep(Xtr_cell_s, Xte_cell_s, C, f"LR cell C={C}"))

# ── blend + adaptive calibration ──────────────────────────────────────────────
all_preds = base_preds + nys_preds + poly_preds + mlp_preds + raw_preds  # 26 models
blend = np.mean(all_preds, axis=0)

base_rates = Ytr.mean(axis=0)
alpha = np.where(base_rates < 0.001, 0.12,
        np.where(base_rates < 0.003, 0.08,
        np.where(base_rates < 0.010, 0.06, 0.04)))

calibrated = blend * (1 - alpha) + base_rates * alpha
calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)

# ── submission ─────────────────────────────────────────────────────────────────
trt_df  = pd.DataFrame(calibrated, columns=target_cols)
trt_df.insert(0, "sig_id", Xte["sig_id"].values)
ctrl_df = pd.DataFrame(
    np.full((int(is_ctrl_te.sum()), len(target_cols)), 0.001), columns=target_cols)
ctrl_df.insert(0, "sig_id", test_features[is_ctrl_te]["sig_id"].values)

sub = pd.concat([trt_df, ctrl_df], ignore_index=True)
sub = sub.set_index("sig_id").loc[test_features["sig_id"].values].reset_index()
sub.to_csv("submission.csv", index=False)
print(f"Done — {len(sub)} rows × {len(target_cols)} targets  ({time.time()-t0:.1f}s total)")
