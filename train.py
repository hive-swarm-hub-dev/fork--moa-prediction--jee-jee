"""
MoA Prediction — LogReg + Kernel Approximation ensemble
Strategy: avoid MLPs (too slow to converge on this CPU); instead use
  1. LogReg on rich PCA + interaction features (baseline linear model)
  2. LogReg on RBF kernel approximation (non-linear, RBF-SVM equivalent)
  3. LogReg on explicit degree-2 interactions of top PCA components

All LogReg models are well-calibrated. Per-target adaptive Bayesian
calibration provides final correction toward training base rates.

Sets OMP_NUM_THREADS=1 before imports so loky parallel workers don't
fight over BLAS threads.
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
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
t0 = time.time()

# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
train_features = pd.read_csv("data/train_features.csv")
train_targets  = pd.read_csv("data/train_targets.csv")
test_features  = pd.read_csv("data/test_features.csv")
target_cols = [c for c in train_targets.columns if c != "sig_id"]

is_ctrl_tr = train_features["cp_type"] == "ctl_vehicle"
is_ctrl_te = test_features["cp_type"]  == "ctl_vehicle"
Xtr = train_features[~is_ctrl_tr].copy().reset_index(drop=True)
Ytr = train_targets[~is_ctrl_tr][target_cols].values.astype(np.float32)
Xte = test_features[~is_ctrl_te].copy().reset_index(drop=True)

gene_cols = [c for c in train_features.columns if c.startswith("g-")]
cell_cols  = [c for c in train_features.columns if c.startswith("c-")]

# ── feature engineering ────────────────────────────────────────────────────────
print("Fitting transformers...")
gsc  = StandardScaler().fit(Xtr[gene_cols])
csc  = StandardScaler().fit(Xtr[cell_cols])
gcsc = StandardScaler().fit(Xtr[gene_cols + cell_cols])

g_pca  = PCA(n_components=150, random_state=42).fit(gsc.transform(Xtr[gene_cols]))
c_pca  = PCA(n_components=50,  random_state=42).fit(csc.transform(Xtr[cell_cols]))
gc_pca = PCA(n_components=40,  random_state=42).fit(gcsc.transform(Xtr[gene_cols + cell_cols]))

# top-variance raw features (extra signal for rare targets)
top_g_cols = Xtr[gene_cols].var().nlargest(30).index.tolist()
top_c_cols = Xtr[cell_cols].var().nlargest(20).index.tolist()
tgsc = StandardScaler().fit(Xtr[top_g_cols])
tcsc = StandardScaler().fit(Xtr[top_c_cols])

def make_features(df):
    g  = gsc.transform(df[gene_cols])
    c  = csc.transform(df[cell_cols])
    gc = gcsc.transform(df[gene_cols + cell_cols])
    gp  = g_pca.transform(g)    # (n, 150)
    cp_ = c_pca.transform(c)    # (n, 50)
    gcp = gc_pca.transform(gc)  # (n, 40)

    t    = df["cp_time"].values.astype(float)
    dose = (df["cp_dose"] == "D2").astype(float).values
    tnorm = (t - 24) / 48
    t_oh  = np.column_stack([(t == v).astype(float) for v in [24, 48, 72]])

    # gene-PCA × condition interactions (top-20 gene PCs)
    ints = np.hstack([
        gp[:, :20] * tnorm[:, None],
        gp[:, :20] * dose[:, None],
        gp[:, :20] * (tnorm * dose)[:, None],
    ])  # (n, 60)

    # cross-PCA: top-5 gene PC × top-5 cell PC
    cross = (gp[:, :5, None] * cp_[:, None, :5]).reshape(len(df), 25)

    # per-row statistics
    raw_g = df[gene_cols].values
    raw_c = df[cell_cols].values
    g_kurt = kurtosis_fn(raw_g, axis=1).reshape(-1, 1)
    c_kurt = kurtosis_fn(raw_c, axis=1).reshape(-1, 1)
    g_iqr  = (np.percentile(raw_g, 75, axis=1) - np.percentile(raw_g, 25, axis=1)).reshape(-1, 1)
    c_iqr  = (np.percentile(raw_c, 75, axis=1) - np.percentile(raw_c, 25, axis=1)).reshape(-1, 1)

    tg   = tgsc.transform(df[top_g_cols])
    tc   = tcsc.transform(df[top_c_cols])
    cond = np.column_stack([tnorm, dose, tnorm * dose, t_oh])

    return np.hstack([gp, cp_, gcp, ints, cross,
                      g_kurt, c_kurt, g_iqr, c_iqr,
                      tg, tc, cond]).astype(np.float32)

print("Building features...")
Xtr_f = make_features(Xtr)
Xte_f = make_features(Xte)
print(f"  base dim={Xtr_f.shape[1]}  ({time.time()-t0:.1f}s)")

fsc   = StandardScaler().fit(Xtr_f)
Xtr_s = fsc.transform(Xtr_f)
Xte_s = fsc.transform(Xte_f)

# RBF kernel features for non-linearity (multiple gammas)
rbf_maps = []
for gamma in [0.01, 0.1, 1.0]:
    rbf = RBFSampler(gamma=gamma, n_components=500, random_state=42).fit(Xtr_s)
    rbf_maps.append(rbf)
    print(f"  RBF gamma={gamma}: {time.time()-t0:.1f}s")

# Explicit degree-2 interactions on top-20 PCA components + conditions
# Top-20 gene PCA + 3 conditions = 23 features → 23×22/2 = 253 interaction pairs
top_feats = Xtr_s[:, :20]   # top-20 gene PCA components (after StandardScaler)
top_te    = Xte_s[:, :20]
n = top_feats.shape[0]
pairs_tr = np.column_stack([top_feats[:, i] * top_feats[:, j]
                             for i in range(20) for j in range(i+1, 20)])
pairs_te = np.column_stack([top_te[:, i] * top_te[:, j]
                             for i in range(20) for j in range(i+1, 20)])
poly_tr = np.hstack([Xtr_s, pairs_tr]).astype(np.float32)
poly_te = np.hstack([Xte_s, pairs_te]).astype(np.float32)
poly_fsc = StandardScaler().fit(poly_tr)
poly_tr = poly_fsc.transform(poly_tr)
poly_te = poly_fsc.transform(poly_te)
print(f"  poly dim={poly_tr.shape[1]}  ({time.time()-t0:.1f}s)")

# ── parallel per-target fitting ────────────────────────────────────────────────
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
        # Only class 0 seen in training — positive class has zero probability
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

# warm up loky pool
print("Warming up worker pool...")
Parallel(n_jobs=-1)([delayed(lambda: 1)() for _ in range(48)])

# 1. Base LogReg (three C values)
print(f"Base LogReg sweeps... ({time.time()-t0:.1f}s elapsed)")
base_preds = []
for C in [0.05, 0.10, 0.20]:
    base_preds.append(sweep(Xtr_s, Xte_s, C, f"LR base C={C}"))

# 2. RBF kernel LogReg (three gammas)
print(f"RBF kernel LogReg... ({time.time()-t0:.1f}s elapsed)")
rbf_preds = []
for gamma, rbf in zip([0.01, 0.1, 1.0], rbf_maps):
    Xtr_rbf = rbf.transform(Xtr_s).astype(np.float32)
    Xte_rbf = rbf.transform(Xte_s).astype(np.float32)
    rbf_preds.append(sweep(Xtr_rbf, Xte_rbf, 0.1, f"LR RBF gamma={gamma}"))

# 3. Polynomial feature LogReg
print(f"Polynomial LogReg... ({time.time()-t0:.1f}s elapsed)")
poly_preds = [sweep(poly_tr, poly_te, 0.1, "LR poly")]

# ── blend all models ───────────────────────────────────────────────────────────
all_preds = base_preds + rbf_preds + poly_preds   # 7 models
blend = np.mean(all_preds, axis=0)

# per-target adaptive Bayesian calibration
base_rates = Ytr.mean(axis=0)
alpha = np.where(base_rates < 0.001, 0.12,
        np.where(base_rates < 0.003, 0.08,
        np.where(base_rates < 0.010, 0.06, 0.04)))

calibrated = blend * (1 - alpha) + base_rates * alpha
calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)

# ── submission ─────────────────────────────────────────────────────────────────
trt_df = pd.DataFrame(calibrated, columns=target_cols)
trt_df.insert(0, "sig_id", Xte["sig_id"].values)
ctrl_df = pd.DataFrame(
    np.full((int(is_ctrl_te.sum()), len(target_cols)), 0.001), columns=target_cols)
ctrl_df.insert(0, "sig_id", test_features[is_ctrl_te]["sig_id"].values)

sub = pd.concat([trt_df, ctrl_df], ignore_index=True)
sub = sub.set_index("sig_id").loc[test_features["sig_id"].values].reset_index()
sub.to_csv("submission.csv", index=False)
print(f"Done — {len(sub)} rows × {len(target_cols)} targets  ({time.time()-t0:.1f}s total)")
