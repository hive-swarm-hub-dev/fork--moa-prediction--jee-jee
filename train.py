"""
MoA Prediction — Rich feature LogReg ensemble v2
Features match chanbin-test-moa's best (af3682d5):
  - PCA(200 gene + 60 cell + 50 combined)
  - Gene-PCA × cp_time/cp_dose interaction features (top-20 PCs)
  - Cross-PCA: top-5 gene × top-5 cell
  - Per-row statistics: kurtosis, percentiles, IQR
  - Top-30 raw high-variance gene features
Improvements:
  - Per-target adaptive Bayesian calibration
  - Nystroem kernel model (better kernel approximation than RBFSampler)
  - 5 LogReg C-values for better diversity
  - Fix: DummyClassifier class ordering bug (critical for all-zero columns)
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

# Match chanbin's best PCA settings
g_pca  = PCA(n_components=200, random_state=42).fit(gsc.transform(Xtr[gene_cols]))
c_pca  = PCA(n_components=60,  random_state=42).fit(csc.transform(Xtr[cell_cols]))
gc_pca = PCA(n_components=50,  random_state=42).fit(gcsc.transform(Xtr[gene_cols + cell_cols]))

# Top-variance raw features
top_g_cols = Xtr[gene_cols].var().nlargest(30).index.tolist()
top_c_cols = Xtr[cell_cols].var().nlargest(20).index.tolist()
tgsc = StandardScaler().fit(Xtr[top_g_cols])
tcsc = StandardScaler().fit(Xtr[top_c_cols])

def make_features(df):
    g  = gsc.transform(df[gene_cols])
    c  = csc.transform(df[cell_cols])
    gc = gcsc.transform(df[gene_cols + cell_cols])
    gp  = g_pca.transform(g)    # (n, 200)
    cp_ = c_pca.transform(c)    # (n, 60)
    gcp = gc_pca.transform(gc)  # (n, 50)

    t    = df["cp_time"].values.astype(float)
    dose = (df["cp_dose"] == "D2").astype(float).values
    tnorm = (t - 24) / 48
    t_oh  = np.column_stack([(t == v).astype(float) for v in [24, 48, 72]])

    # gene-PCA × condition interactions (top-20 gene PCs = 60 features)
    ints = np.hstack([
        gp[:, :20] * tnorm[:, None],
        gp[:, :20] * dose[:, None],
        gp[:, :20] * (tnorm * dose)[:, None],
    ])

    # cross-PCA: top-5 gene PC × top-5 cell PC (25 features)
    cross = (gp[:, :5, None] * cp_[:, None, :5]).reshape(len(df), 25)

    # per-row gene statistics (kurtosis + 5 percentiles + IQR = 8)
    raw_g = df[gene_cols].values
    raw_c = df[cell_cols].values
    g_kurt = kurtosis_fn(raw_g, axis=1).reshape(-1, 1)
    c_kurt = kurtosis_fn(raw_c, axis=1).reshape(-1, 1)
    g_pct  = np.percentile(raw_g, [10, 25, 50, 75, 90], axis=1).T  # (n, 5)
    c_pct  = np.percentile(raw_c, [10, 25, 50, 75, 90], axis=1).T  # (n, 5)
    g_iqr  = (g_pct[:, 3] - g_pct[:, 1]).reshape(-1, 1)
    c_iqr  = (c_pct[:, 3] - c_pct[:, 1]).reshape(-1, 1)

    tg   = tgsc.transform(df[top_g_cols])   # (n, 30)
    tc   = tcsc.transform(df[top_c_cols])   # (n, 20)
    cond = np.column_stack([tnorm, dose, tnorm * dose, t_oh])  # (n, 6)

    return np.hstack([gp, cp_, gcp, ints, cross,
                      g_kurt, c_kurt, g_pct, c_pct, g_iqr, c_iqr,
                      tg, tc, cond]).astype(np.float32)

print("Building features...")
Xtr_f = make_features(Xtr)
Xte_f = make_features(Xte)
print(f"  base dim={Xtr_f.shape[1]}  ({time.time()-t0:.1f}s)")

fsc   = StandardScaler().fit(Xtr_f)
Xtr_s = fsc.transform(Xtr_f)
Xte_s = fsc.transform(Xte_f)

# Nystroem kernel approximation (non-linear model via kernel trick)
print("Fitting Nystroem kernel...")
nys = Nystroem(kernel='rbf', gamma=0.05, n_components=300, random_state=42).fit(Xtr_s)
Xtr_nys = nys.transform(Xtr_s).astype(np.float32)
Xte_nys = nys.transform(Xte_s).astype(np.float32)
nys_fsc = StandardScaler().fit(Xtr_nys)
Xtr_nys = nys_fsc.transform(Xtr_nys)
Xte_nys = nys_fsc.transform(Xte_nys)
print(f"  Nystroem dim={Xtr_nys.shape[1]}  ({time.time()-t0:.1f}s)")

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
        return np.zeros(len(X_te))   # all-zero training → predict 0 for positive class
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

# ── blend + adaptive calibration ──────────────────────────────────────────────
all_preds = base_preds + nys_preds   # 7 models
blend = np.mean(all_preds, axis=0)

base_rates = Ytr.mean(axis=0)
# Per-target adaptive Bayesian calibration (stronger for rarer targets)
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
