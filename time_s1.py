"""Quick timing test for Stage 1 sweeps."""
import os, time
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings; warnings.filterwarnings('ignore')

t0 = time.time()
print("Loading data...", flush=True)
train_features = pd.read_csv("data/train_features.csv")
train_targets   = pd.read_csv("data/train_targets.csv")
test_features   = pd.read_csv("data/test_features.csv")
target_cols = [c for c in train_targets.columns if c != "sig_id"]

is_ctrl_tr = train_features["cp_type"] == "ctl_vehicle"
Xtr = train_features[~is_ctrl_tr].copy().reset_index(drop=True)
Ytr = train_targets[~is_ctrl_tr][target_cols].values.astype(np.float32)

gene_cols = [c for c in train_features.columns if c.startswith("g-")]
cell_cols  = [c for c in train_features.columns if c.startswith("c-")]
print(f"  {len(Xtr)} treatment samples, {len(target_cols)} targets  ({time.time()-t0:.1f}s)", flush=True)

print("PCA...", flush=True)
all_feats = pd.concat([train_features, test_features], ignore_index=True)
pca_g  = PCA(n_components=120, whiten=True, random_state=42)
pca_c  = PCA(n_components=50,  whiten=True, random_state=42)
pca_gc = PCA(n_components=50,  whiten=True, random_state=42)
g_all  = pca_g.fit_transform(all_feats[gene_cols].values)
c_all  = pca_c.fit_transform(all_feats[cell_cols].values)
gc_all = pca_gc.fit_transform(all_feats[gene_cols + cell_cols].values)
n_train = len(train_features); trt_mask = ~is_ctrl_tr.values
g_tr = g_all[:n_train][trt_mask]; c_tr = c_all[:n_train][trt_mask]; gc_tr = gc_all[:n_train][trt_mask]
print(f"  PCA done  ({time.time()-t0:.1f}s)", flush=True)

# Simple features (no kurtosis/percentile for speed in this test)
Xtr_s = StandardScaler().fit_transform(
    np.hstack([g_tr, c_tr, gc_tr]).astype(np.float32)
)  # 220-dim
print(f"  Features 220-dim  ({time.time()-t0:.1f}s)", flush=True)
n_targets = len(target_cols)

def fit_lr(X_tr, y_col, C):
    y = y_col.astype(int)
    if len(np.unique(y)) < 2:
        return DummyClassifier(strategy="most_frequent").fit(X_tr, y)
    return LogisticRegression(C=C, max_iter=500, solver='liblinear').fit(X_tr, y)

def fit_ridge(X_tr, y_col):
    return Ridge(alpha=1.0).fit(X_tr, y_col.astype(float))

# Time one LR sweep (full data, threading)
print("\n--- Timing LR sweeps (threading, full data 17560 samples, 220-dim) ---", flush=True)
for C in [0.05, 0.10, 0.20, 0.50]:
    t1 = time.time()
    ests = Parallel(n_jobs=-1, prefer='threads')(
        delayed(fit_lr)(Xtr_s, Ytr[:, j], C) for j in range(n_targets))
    print(f"  LR C={C}: {time.time()-t1:.2f}s", flush=True)

t1 = time.time()
ests = Parallel(n_jobs=-1, prefer='threads')(
    delayed(fit_ridge)(Xtr_s, Ytr[:, j]) for j in range(n_targets))
print(f"  Ridge: {time.time()-t1:.2f}s", flush=True)

# Time on half data (OOF scenario)
kf = KFold(n_splits=2, shuffle=True, random_state=42)
tr_idx, val_idx = next(iter(kf.split(Xtr_s)))
X_half = Xtr_s[tr_idx]
y_half = Ytr[tr_idx]
print(f"\n--- Half data ({len(X_half)} samples) ---", flush=True)
for C in [0.05, 0.10, 0.20, 0.50]:
    t1 = time.time()
    ests = Parallel(n_jobs=-1, prefer='threads')(
        delayed(fit_lr)(X_half, y_half[:, j], C) for j in range(n_targets))
    print(f"  LR C={C}: {time.time()-t1:.2f}s", flush=True)

t1 = time.time()
ests = Parallel(n_jobs=-1, prefer='threads')(
    delayed(fit_ridge)(X_half, y_half[:, j]) for j in range(n_targets))
print(f"  Ridge: {time.time()-t1:.2f}s", flush=True)

print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
