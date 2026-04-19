"""
metrics.py — Evaluation metrics and synthetic data generators.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation


# ──────────────── Overlap & Boundary Metrics ────────────────

def dice(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    return float(2*np.logical_and(p,g).sum()/(p.sum()+g.sum()+1e-12))

def dice_multiclass(pred, gt, K):
    return {k: dice(pred==k, gt==k) for k in range(K)}

def hd95(pred, gt):
    ps = np.logical_xor(pred.astype(bool), binary_erosion(pred.astype(bool)))
    gs = np.logical_xor(gt.astype(bool), binary_erosion(gt.astype(bool)))
    if ps.sum()==0 and gs.sum()==0: return 0.0
    if ps.sum()==0 or gs.sum()==0: return float('inf')
    d1 = distance_transform_edt(~gs)[ps]; d2 = distance_transform_edt(~ps)[gs]
    return float(np.percentile(np.concatenate([d1,d2]), 95))

def assd(pred, gt):
    ps = np.logical_xor(pred.astype(bool), binary_erosion(pred.astype(bool)))
    gs = np.logical_xor(gt.astype(bool), binary_erosion(gt.astype(bool)))
    if ps.sum()==0 and gs.sum()==0: return 0.0
    if ps.sum()==0 or gs.sum()==0: return float('inf')
    return float((distance_transform_edt(~gs)[ps].mean() + distance_transform_edt(~ps)[gs].mean())/2)

def surface_dice(pred, gt, tol=2.0):
    ps = np.logical_xor(pred.astype(bool), binary_erosion(pred.astype(bool)))
    gs = np.logical_xor(gt.astype(bool), binary_erosion(gt.astype(bool)))
    if ps.sum()==0 and gs.sum()==0: return 1.0
    if ps.sum()==0 or gs.sum()==0: return 0.0
    return float(((distance_transform_edt(~gs)[ps]<=tol).sum()+(distance_transform_edt(~ps)[gs]<=tol).sum())/(ps.sum()+gs.sum()+1e-12))

# ──────────────── Calibration ────────────────

def ece(probs, labels, n_bins=15):
    p, l = probs.ravel(), labels.ravel().astype(float)
    bins = np.linspace(0,1,n_bins+1)
    accs = np.zeros(n_bins); confs = np.zeros(n_bins); cnts = np.zeros(n_bins)
    for i in range(n_bins):
        m = (p>bins[i])&(p<=bins[i+1])
        if m.sum()>0: accs[i]=l[m].mean(); confs[i]=p[m].mean(); cnts[i]=m.sum()
    return float(np.sum(cnts/p.size*np.abs(accs-confs))), accs, confs, cnts

def brier(probs, labels):
    return float(np.mean((probs.ravel()-labels.ravel().astype(float))**2))

def entropy_map(probs, eps=1e-12):
    p = np.clip(probs, eps, 1-eps)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def find_optimal_temperature(probs, labels, t_range=(0.1,5.0), n_steps=100):
    logits = np.log(np.clip(probs,1e-8,1-1e-8)/(1-np.clip(probs,1e-8,1-1e-8)))
    best_t, best_e = 1.0, float('inf')
    for t in np.linspace(t_range[0], t_range[1], n_steps):
        scaled = 1.0/(1.0+np.exp(-logits/t))
        e, _, _, _ = ece(scaled, labels)
        if e < best_e: best_e = e; best_t = t
    return best_t


# ──────────────── Synthetic Data ────────────────

def make_phantom(shape=(128,128), center=None, radii=(35,25), noise_std=0.08, seed=42):
    H, W = shape
    if center is None: center = (H//2, W//2)
    yy, xx = np.ogrid[:H,:W]
    gt = (((yy-center[0])/radii[0])**2 + ((xx-center[1])/radii[1])**2 <= 1).astype(np.int32)
    rng = np.random.default_rng(seed)
    img = gt.astype(float)*0.7 + noise_std*rng.standard_normal(shape)
    return gt, img

def make_raters(gt, n_raters=5, sens_range=(0.75,0.95), spec_range=(0.90,0.99),
                jitter=3, seed=42, include_outlier=False):
    rng = np.random.default_rng(seed)
    H,W = gt.shape
    sens = rng.uniform(sens_range[0], sens_range[1], n_raters)
    spec = rng.uniform(spec_range[0], spec_range[1], n_raters)
    R = np.zeros((n_raters,H,W), dtype=np.int32)
    for a in range(n_raters):
        m = gt.copy()
        if jitter > 0:
            j = rng.integers(1, jitter+1)
            struct = np.ones((2*j+1, 2*j+1))
            m = (binary_dilation(m, structure=struct) if rng.random()>0.5 
                 else binary_erosion(m, structure=struct)).astype(np.int32)
        noisy = m.copy()
        noisy[gt==1] = np.where(rng.random(gt.shape)[gt==1]>sens[a], 0, m[gt==1])
        noisy[gt==0] = np.where(rng.random(gt.shape)[gt==0]>spec[a], 1, m[gt==0])
        R[a] = noisy
    if include_outlier:
        correct = rng.random(gt.shape) < 0.5
        R[-1] = np.where(correct, gt, 1-gt)
        sens[-1] = 0.5; spec[-1] = 0.5
    return R, sens, spec

def make_spatial_raters(gt, n_raters=6, seed=42):
    """Raters with spatially varying quality — left half vs right half."""
    rng = np.random.default_rng(seed)
    H, W = gt.shape; mid = W//2
    R = np.zeros((n_raters, H, W), dtype=np.int32)
    for a in range(n_raters):
        noisy = gt.copy()
        if a < n_raters//2:
            sl, sr = 0.95, 0.60; tl, tr = 0.98, 0.80
        else:
            sl, sr = 0.60, 0.95; tl, tr = 0.80, 0.98
        for mask, s, t in [(np.s_[:, :mid], sl, tl), (np.s_[:, mid:], sr, tr)]:
            fg = gt[mask]==1; bg = gt[mask]==0
            region = noisy[mask]
            region[fg] = np.where(rng.random(fg.sum())>s, 0, 1)
            region[bg] = np.where(rng.random(bg.sum())>t, 1, 0)
            noisy[mask] = region
        j = rng.integers(1, 3); struct = np.ones((2*j+1,2*j+1))
        if rng.random()>0.5:
            noisy = binary_dilation(noisy, structure=struct).astype(np.int32)
        R[a] = noisy
    return R

def make_multiclass_phantom(shape=(128,128), K=4, seed=42):
    H,W = shape; cy,cx = H//2, W//2
    gt = np.zeros(shape, dtype=np.int32)
    yy,xx = np.ogrid[:H,:W]
    if K>=2: gt[((yy-cy)/(H//3))**2+((xx-cx)/(W//3))**2<=1] = 1
    if K>=3: gt[((yy-cy)/(H//5))**2+((xx-cx)/(W//5))**2<=1] = 2
    if K>=4: gt[((yy-cy+5)/(H//10))**2+((xx-cx)/(W//10))**2<=1] = 3
    return gt

def make_multiclass_raters(gt, K, n_raters=5, diag_range=(0.7,0.95), seed=42):
    rng = np.random.default_rng(seed)
    N = gt.size; gf = gt.ravel()
    true_C = np.zeros((n_raters, K, K))
    R = np.zeros((n_raters, N), dtype=np.int32)
    for a in range(n_raters):
        C = np.zeros((K,K))
        for i in range(K):
            d = rng.uniform(diag_range[0], diag_range[1])
            C[i,i] = d
            off = rng.dirichlet(np.ones(K-1))*(1-d)
            idx = 0
            for j in range(K):
                if j!=i: C[i,j] = off[idx]; idx += 1
        true_C[a] = C
        for v in range(N):
            R[a,v] = rng.choice(K, p=C[gf[v]])
    return R.reshape(n_raters, *gt.shape), true_C
