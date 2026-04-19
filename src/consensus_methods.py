"""
consensus_methods.py — Complete consensus segmentation library.

Implements:
  1. Majority Vote (MV)
  2. STAPLE — binary & multi-class (Warfield et al., IEEE TMI 2004)
  3. Spatial STAPLE — spatially varying performance fields (Asman & Landman, IEEE TMI 2012)
  4. SIMPLE-like boundary-weighted fusion (inspired by Langerak et al., IEEE TMI 2010)
  5. Hybrid STAPLE-interior + SIMPLE-boundary
  6. Log-opinion pooling

All methods accept (A, *spatial) arrays and return dicts with 'consensus' and 'posterior'.
"""

import numpy as np
from scipy.ndimage import (
    gaussian_gradient_magnitude, uniform_filter, gaussian_filter,
    binary_dilation, binary_erosion
)

# ─────────────────────── Utilities ───────────────────────

def _ls(x, eps=1e-12):
    """Safe log."""
    return np.log(np.clip(x, eps, None))


# ─────────────────────── 1. Majority Vote ───────────────────────

def majority_vote(R):
    """Simple majority vote. R: (A, *spatial), binary."""
    return (R.sum(0) > R.shape[0] / 2.0).astype(np.int32)


def majority_vote_multiclass(R, K):
    """Multi-class majority vote. R: (A, *spatial), labels 0..K-1."""
    N = np.prod(R.shape[1:])
    Y = R.reshape(-1, N)
    votes = np.zeros((K, N))
    for k in range(K):
        votes[k] = (Y == k).sum(0)
    return np.argmax(votes, 0).reshape(R.shape[1:]).astype(np.int32)


# ─────────────────────── 2. STAPLE ───────────────────────

def staple_binary(R, prior=0.5, max_iter=50, tol=1e-6, damping=0.0,
                  alpha_s=(1,1), alpha_t=(1,1), exclude_consensus=False):
    """
    Binary STAPLE (Warfield et al. 2004).
    
    When exclude_consensus=True, consensus voxels (all raters agree) are 
    fixed and only non-consensus voxels participate in EM — following
    Rohlfing et al. 2004 / Van Leemput & Sabuncu 2014 "Restricted" variant.
    """
    A = R.shape[0]; sh = R.shape[1:]; N = np.prod(sh)
    Y = R.reshape(A, N).astype(np.float64)
    
    # Identify consensus voxels
    if exclude_consensus:
        all_agree = np.all(Y == Y[0:1], axis=0)
        consensus_val = Y[0][all_agree]
    else:
        all_agree = np.zeros(N, dtype=bool)
    
    p = np.clip(Y.mean(0), 1e-4, 1-1e-4)
    s = np.full(A, 0.9999); t = np.full(A, 0.9999)
    ll_trace = []
    
    for it in range(max_iter):
        # E-step
        lp1 = np.full(N, _ls(np.array([prior]))[0])
        lp0 = np.full(N, _ls(np.array([1-prior]))[0])
        for a in range(A):
            lp1 += Y[a]*_ls(s[a]) + (1-Y[a])*_ls(1-s[a])
            lp0 += Y[a]*_ls(1-t[a]) + (1-Y[a])*_ls(t[a])
        p_new = np.clip(1.0/(1.0+np.exp(lp0-lp1)), 1e-8, 1-1e-8)
        
        if exclude_consensus:
            p_new[all_agree] = consensus_val  # fix consensus voxels
        
        lmax = np.maximum(lp1, lp0)
        ll = np.sum(lmax + np.log(np.exp(lp1-lmax)+np.exp(lp0-lmax)))
        ll_trace.append(ll)
        p = p_new
        
        # M-step
        sp = p.sum(); s1p = (1-p).sum()
        sn = np.array([(p*Y[a]).sum()+alpha_s[0]-1 for a in range(A)]) / (sp+alpha_s[0]+alpha_s[1]-2+1e-12)
        tn = np.array([((1-p)*(1-Y[a])).sum()+alpha_t[0]-1 for a in range(A)]) / (s1p+alpha_t[0]+alpha_t[1]-2+1e-12)
        sn = np.clip(sn, 1e-4, 1-1e-4); tn = np.clip(tn, 1e-4, 1-1e-4)
        if damping > 0:
            sn = (1-damping)*sn + damping*s; tn = (1-damping)*tn + damping*t
        s, t = sn, tn
        
        if it > 0 and abs(ll_trace[-1]-ll_trace[-2]) < tol:
            break
    
    return {'posterior': p.reshape(sh), 'consensus': (p>=0.5).astype(np.int32).reshape(sh),
            'sensitivity': s, 'specificity': t, 'log_likelihood': ll_trace, 'n_iter': it+1}


def staple_multiclass(R, K, max_iter=50, tol=1e-6, damping=0.3, dirichlet_alpha=1.01):
    """Multi-class STAPLE with full K×K confusion matrices."""
    A = R.shape[0]; sh = R.shape[1:]; N = np.prod(sh)
    Y = R.reshape(A, N)
    
    q = np.zeros((N, K))
    for k in range(K): q[:, k] = (Y == k).sum(0)
    q = q / (q.sum(1, keepdims=True) + 1e-12)
    q = np.clip(q, 1e-6, 1-1e-6)
    
    C = np.zeros((A, K, K))
    for a in range(A): C[a] = np.eye(K)*0.9 + 0.1/K
    
    ap = np.ones((K,K))*(dirichlet_alpha-1.0)
    for k in range(K): ap[k,k] = (dirichlet_alpha-1.0)*K
    ll_trace = []
    
    for it in range(max_iter):
        log_q = np.zeros((N, K))
        for k in range(K):
            log_q[:,k] = _ls(1.0/K)
            for a in range(A): log_q[:,k] += _ls(C[a,k,Y[a]])
        lm = log_q.max(1, keepdims=True)
        qu = np.exp(log_q - lm)
        q = qu / (qu.sum(1, keepdims=True) + 1e-12)
        q = np.clip(q, 1e-8, 1-1e-8)
        ll = np.sum(lm.ravel() + np.log(qu.sum(1) + 1e-12))
        ll_trace.append(ll)
        
        Cn = np.zeros((A, K, K))
        for a in range(A):
            for i in range(K):
                for j in range(K):
                    Cn[a,i,j] = (q[:,i]*(Y[a]==j)).sum() + ap[i,j]
            Cn[a] = Cn[a] / (Cn[a].sum(1, keepdims=True) + 1e-12)
        if damping > 0: Cn = (1-damping)*Cn + damping*C
        C = Cn
        
        if it > 0 and abs(ll_trace[-1]-ll_trace[-2]) < tol: break
    
    post = q.reshape(*sh, K)
    return {'posterior': post, 'consensus': np.argmax(post, -1).astype(np.int32),
            'confusion': C, 'log_likelihood': ll_trace, 'n_iter': it+1}


# ─────────────────────── 3. Spatial STAPLE ───────────────────────

def spatial_staple_binary(R, window_frac=0.15, overlap=0.5, prior=0.5,
                          max_iter=30, tol=1e-5, kappa=1.0, smooth_sigma=2.0):
    """
    Spatial STAPLE (Asman & Landman 2012).
    Estimates per-window confusion matrices, regularized by global prior,
    and interpolated via Gaussian smoothing.
    """
    A = R.shape[0]; H, W = R.shape[1:]
    Y = R.astype(np.float64)
    p = np.clip(Y.mean(0), 1e-4, 1-1e-4)
    
    # Global STAPLE as prior
    gl = staple_binary(R, prior=prior, max_iter=20, damping=0.3)
    s0 = gl['sensitivity']; t0 = gl['specificity']
    
    wh = max(int(H*window_frac), 3); ww = max(int(W*window_frac), 3)
    sh = max(int(wh*(1-overlap)), 1); sw = max(int(ww*(1-overlap)), 1)
    sf = np.tile(s0[:,None,None], (1,H,W))
    tf = np.tile(t0[:,None,None], (1,H,W))
    
    for it in range(max_iter):
        op = p.copy()
        for ys in range(0, H, sh):
            for xs in range(0, W, sw):
                ye, xe = min(ys+wh, H), min(xs+ww, W)
                sl = (slice(ys,ye), slice(xs,xe))
                pw = p[sl]; nv = pw.size
                for a in range(A):
                    yw = Y[a][sl]
                    sp = pw.sum(); s1p = (1-pw).sum()
                    ls = (pw*yw).sum()/(sp+1e-12)
                    lt = ((1-pw)*(1-yw)).sum()/(s1p+1e-12)
                    sig = kappa if nv > 10 else kappa*10
                    sf[a,sl[0],sl[1]] = (sig*s0[a]+ls*nv)/(sig+nv)
                    tf[a,sl[0],sl[1]] = (sig*t0[a]+lt*nv)/(sig+nv)
        for a in range(A):
            sf[a] = gaussian_filter(sf[a], sigma=smooth_sigma)
            tf[a] = gaussian_filter(tf[a], sigma=smooth_sigma)
        sf = np.clip(sf, 0.01, 0.99); tf = np.clip(tf, 0.01, 0.99)
        
        lp1 = np.full((H,W), _ls(np.array([prior]))[0])
        lp0 = np.full((H,W), _ls(np.array([1-prior]))[0])
        for a in range(A):
            lp1 += Y[a]*_ls(sf[a]) + (1-Y[a])*_ls(1-sf[a])
            lp0 += Y[a]*_ls(1-tf[a]) + (1-Y[a])*_ls(tf[a])
        p = np.clip(1.0/(1.0+np.exp(lp0-lp1)), 1e-8, 1-1e-8)
        if np.mean(np.abs(p-op)) < tol: break
    
    return {'posterior': p, 'consensus': (p>=0.5).astype(np.int32),
            'sens_field': sf, 'spec_field': tf, 'n_iter': it+1}


# ─────────────────────── 4. SIMPLE-like ───────────────────────

def simple_fusion(R, image=None, max_iter=10, patch_radius=3,
                  alpha_local=0.8, alpha_global=0.2, edge_boost=1.5):
    """SIMPLE-inspired boundary-weighted fusion (Langerak et al. 2010)."""
    A = R.shape[0]; sh = R.shape[1:]
    Y = R.astype(np.float64)
    ref = image.astype(np.float64) if image is not None else Y.mean(0)
    grad = gaussian_gradient_magnitude(ref, sigma=1.0)
    thr = np.percentile(grad, 85)
    gate = np.clip(grad/(thr+1e-12), 0, 2.0)
    cp = Y.mean(0); gw = np.ones(A)/A; ks = 2*patch_radius+1
    for it in range(max_iter):
        old = cp.copy()
        ls = np.zeros_like(Y)
        for a in range(A):
            mr = uniform_filter(Y[a], size=ks); mc = uniform_filter(cp, size=ks)
            mrc = uniform_filter(Y[a]*cp, size=ks)
            mr2 = uniform_filter(Y[a]**2, size=ks); mc2 = uniform_filter(cp**2, size=ks)
            vr = np.clip(mr2-mr**2, 1e-12, None); vc = np.clip(mc2-mc**2, 1e-12, None)
            ncc = (mrc-mr*mc)/(np.sqrt(vr*vc)+1e-12)
            ls[a] = np.clip((ncc+1)/2, 0, 1)
        for a in range(A): gw[a] = np.mean(ls[a])
        ws = gw.sum()
        if ws > 1e-12: gw /= ws
        wsum = np.zeros(sh, dtype=np.float64); wtot = np.zeros(sh, dtype=np.float64)
        for a in range(A):
            w = alpha_local*ls[a] + alpha_global*gw[a]
            w *= (1.0+(edge_boost-1.0)*gate)
            wsum += w*Y[a]; wtot += w
        cp = np.clip(wsum/(wtot+1e-12), 0, 1)
        if np.mean(np.abs(cp-old)) < 1e-5: break
    return {'consensus': (cp>=0.5).astype(np.int32), 'probability_map': cp}


# ─────────────────────── 5. Hybrid ───────────────────────

def hybrid_fusion(R, image, boundary_width=3):
    """STAPLE for interior + SIMPLE at boundaries."""
    st = staple_binary(R, max_iter=50, damping=0.4, alpha_s=(20,5), alpha_t=(20,5))
    si = simple_fusion(R, image=image)
    init = majority_vote(R)
    dilated = binary_dilation(init, iterations=boundary_width)
    eroded = binary_erosion(init, iterations=boundary_width)
    band = dilated.astype(bool) & ~eroded.astype(bool)
    hp = st['posterior'].copy()
    hp[band] = si['probability_map'][band]
    return {'consensus': (hp>=0.5).astype(np.int32), 'posterior': hp, 'boundary_band': band}


# ─────────────────────── 6. Log-Opinion Pooling ───────────────────────

def log_opinion_pool(R, weights=None, temperature=1.0, prior=0.5):
    """
    Logarithmic opinion pooling.
    p_consensus(v) ∝ prior * prod_a p_a(v)^{w_a/T}
    """
    A = R.shape[0]
    Y = R.astype(np.float64)
    if weights is None:
        weights = np.ones(A) / A
    
    log_p = np.log(np.clip(prior, 1e-8, 1-1e-8))
    log_1mp = np.log(np.clip(1-prior, 1e-8, 1-1e-8))
    for a in range(A):
        w = weights[a] / temperature
        log_p += w * _ls(Y[a] * 0.9 + (1-Y[a]) * 0.1)
        log_1mp += w * _ls((1-Y[a]) * 0.9 + Y[a] * 0.1)
    
    p = 1.0 / (1.0 + np.exp(log_1mp - log_p))
    p = np.clip(p, 1e-8, 1-1e-8)
    return {'consensus': (p>=0.5).astype(np.int32), 'posterior': p}
