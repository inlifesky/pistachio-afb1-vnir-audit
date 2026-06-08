"""
Phase B fix - GBM+SG2+SNV bootstrap CI (closes the Table 2 / Figure 4 gap
flagged by external review: every other cross-batch pipeline had a CI;
GBM+SNV did not, simply because its prediction vector was never saved).

Retrains GBM under SG2+SNV on full v1, predicts v3, aggregates to lots,
runs B=2000 stratified bootstrap over the 52 lots at 8/10/15 ppb. Appends
to results/41_bootstrap_CIs.tsv and updates 41_bootstrap_CIs.md.
"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg

RES = r"D:\bioinformatics\project_pistachio_AFB1\results"
B, SEED = 2000, 42

def snv(X):
    X = np.asarray(X, float)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
    return (X - mu) / sd

# Load + prep (matching script 36 V1_SG2_SNV)
X1 = np.load(os.path.join(RES, "pistachio_spectra.npy"))
m1 = pd.read_csv(os.path.join(RES, "pistachio_meta.tsv"), sep="\t")
X3 = np.load(os.path.join(RES, "pistachio_v3_spectra.npy"))
m3 = pd.read_csv(os.path.join(RES, "pistachio_v3_meta.tsv"), sep="\t")
y1 = m1["AFB1_ppb"].values; y3 = m3["AFB1_ppb"].values

X1t = snv(_sg(X1.astype(float), 2))
X3t = snv(_sg(X3.astype(float), 2))

print("[fit] GBM+SG2+SNV on v1", flush=True)
gbm = HistGradientBoostingRegressor(max_iter=200, max_depth=8,
                                     learning_rate=0.1, min_samples_leaf=20,
                                     random_state=42)
gbm.fit(X1t, y1)
pred3 = gbm.predict(X3t)
np.save(os.path.join(RES, "45_pred_v3_GBM_SNV.npy"), pred3)

# lot-level mean
df = m3.copy(); df["pred"] = pred3
lot = df.groupby("image").agg(pred=("pred","mean"), ppb=("AFB1_ppb","first")).reset_index()
print(f"[lot] n={len(lot)}", flush=True)

def fpr_at_100(y, s):
    if y.sum()==0: return np.nan
    return float((s[y==0] >= s[y==1].min()).mean())

def metric(y, s):
    if y.sum() in (0, len(y)): return dict(AUC=np.nan, PRAUC=np.nan, FPR100=np.nan)
    return dict(AUC=roc_auc_score(y,s),
                PRAUC=average_precision_score(y,s),
                FPR100=fpr_at_100(y,s))

rng = np.random.default_rng(SEED)
rows = []
for thr in [8, 10, 15]:
    y = (lot["ppb"].values >= thr).astype(int)
    s = lot["pred"].values
    pt = metric(y, s)
    ipos, ineg = np.where(y==1)[0], np.where(y==0)[0]
    A = np.empty(B); P = np.empty(B); F = np.empty(B)
    for b in range(B):
        bi = np.concatenate([rng.choice(ipos, len(ipos), True),
                              rng.choice(ineg, len(ineg), True)])
        m = metric(y[bi], s[bi])
        A[b]=m["AUC"]; P[b]=m["PRAUC"]; F[b]=m["FPR100"]
    def ci(a):
        a = a[~np.isnan(a)]; return (float(np.percentile(a,2.5)),
                                       float(np.percentile(a,97.5)))
    rows.append(dict(pipeline="GBM_SNV", threshold=thr,
                     n_lot=len(lot), n_pos=int(y.sum()), n_neg=int((1-y).sum()),
                     AUC=pt["AUC"], AUC_lo=ci(A)[0], AUC_hi=ci(A)[1],
                     PRAUC=pt["PRAUC"], PRAUC_lo=ci(P)[0], PRAUC_hi=ci(P)[1],
                     FPR100=pt["FPR100"], FPR100_lo=ci(F)[0], FPR100_hi=ci(F)[1]))
    print(f"  thr={thr}  AUC={pt['AUC']:.3f} [{ci(A)[0]:.3f}, {ci(A)[1]:.3f}]  "
          f"PRAUC={pt['PRAUC']:.3f} [{ci(P)[0]:.3f}, {ci(P)[1]:.3f}]  "
          f"FPR100={pt['FPR100']:.3f} [{ci(F)[0]:.3f}, {ci(F)[1]:.3f}]", flush=True)

new = pd.DataFrame(rows)
old = pd.read_csv(os.path.join(RES, "41_bootstrap_CIs.tsv"), sep="\t")
# remove any prior GBM_SNV row, append new
old = old[old.pipeline != "GBM_SNV"]
merged = pd.concat([old, new], ignore_index=True)
merged.to_csv(os.path.join(RES, "41_bootstrap_CIs.tsv"), sep="\t", index=False)
print("[write] 41_bootstrap_CIs.tsv updated (GBM_SNV appended)", flush=True)
print("DONE", flush=True)
