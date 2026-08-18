"""The reconciliation frame — byte-state space alignment utilities.

Byte language C gives the state inventory; this module holds the
frame math: per-side whitening, orthogonal procrustes (fp64 SVD —
fp32 SVD is a recorded failure class), stratified held-out site
splits (never first-N), the state codebook, and the two frame losses
(identification InfoNCE against the state inventory, frame-pinned
MSE). Reconciliation lives in byte-state space: no loss in this
module ever takes a batch-index label — negatives, when a
contrastive denominator exists, are the OTHER STATES of the
inventory, never other batch items.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------- splits
def split_sites(sid: np.ndarray, seed: int,
                frac: tuple[float, float, float] = (.25, .5, .25)
                ) -> dict[str, np.ndarray]:
    """Stratified per-state site split into fit/train/eval index sets,
    three-way disjoint (whitening fits on 'fit' only). Shuffled within
    each state (never first-N). Count-2 states: 1 train / 1 eval, no
    fit (rare states never feed the whitening; their near-duplicate
    held-out context is a disclosed instrument limit). Count-1: train
    only, excluded from eval and flagged by the caller's census."""
    rng = np.random.default_rng(seed)
    out = {"fit": [], "train": [], "eval": []}
    for s in np.unique(sid):
        ix = np.flatnonzero(sid == s)
        ix = ix[rng.permutation(len(ix))]
        if len(ix) == 1:
            out["train"].extend(ix)
            continue
        if len(ix) == 2:
            out["train"].append(ix[0])
            out["eval"].append(ix[1])
            continue
        n_fit = max(1, int(round(len(ix) * frac[0])))
        n_tr = max(1, int(round(len(ix) * frac[1])))
        if n_fit + n_tr >= len(ix):
            n_fit = max(1, len(ix) - 2)
            n_tr = 1
        out["fit"].extend(ix[:n_fit])
        out["train"].extend(ix[n_fit:n_fit + n_tr])
        out["eval"].extend(ix[n_fit + n_tr:])
    return {k: np.sort(np.array(v, dtype=np.int64))
            for k, v in out.items()}


# ---------------------------------------------------------- whitening
def fit_whitening(x: np.ndarray, eps: float = 1e-3
                  ) -> tuple[np.ndarray, np.ndarray]:
    """(mu, W) such that (x - mu) @ W has ~identity covariance.
    Fit ONLY on the 'fit' split; eigendecomposition in fp64 with
    shrinkage eps toward the mean eigenvalue."""
    x = x.astype(np.float64)
    mu = x.mean(0)
    c = np.cov(x - mu, rowvar=False)
    c = (1 - eps) * c + eps * np.trace(c) / c.shape[0] * np.eye(
        c.shape[0])
    ev, U = np.linalg.eigh(c)
    ev = np.maximum(ev, 1e-10)
    return mu, U @ np.diag(ev ** -0.5) @ U.T


def apply_whitening(x: np.ndarray, mu: np.ndarray,
                    w: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float64) - mu) @ w).astype(np.float32)


# ---------------------------------------------------------- procrustes
def procrustes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Orthogonal map R minimizing ||x @ R - y||_F, SVD in fp64
    (fp32 SVD here is a recorded crash class). x: (n, dx), y: (n, dy)
    with dx >= dy: R is (dx, dy) with orthonormal columns."""
    m = x.astype(np.float64).T @ y.astype(np.float64)
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return (u @ vt).astype(np.float64)


# ------------------------------------------ byte-structural features
def state_byte_features(texts: list[bytes],
                        prev_ctx: np.ndarray | None = None,
                        next_ctx: np.ndarray | None = None
                        ) -> np.ndarray:
    """Model-free per-state feature matrix from the byte language
    itself: byte unigram histogram (256) + hashed byte-bigram
    histogram (256) + optional attested boundary-context profiles
    (preceding/following byte distributions, (n,256) each) + length.
    All histograms L1-normalized so corpus FREQUENCY never enters the
    geometry (the prior belongs in a bias channel, never the frame).
    This is the codebook's ancestry: E_C initializes from C's own
    structure, not from any model."""
    n = len(texts)
    uni = np.zeros((n, 256))
    big = np.zeros((n, 256))
    ln = np.zeros((n, 1))
    for i, t in enumerate(texts):
        for b in t:
            uni[i, b] += 1
        for a, b in zip(t, t[1:]):
            big[i, (a * 31 + b) % 256] += 1
        uni[i] /= max(len(t), 1)
        if len(t) > 1:
            big[i] /= len(t) - 1
        ln[i, 0] = len(t) / 16.0
    parts = [uni, big, ln]
    for ctx in (prev_ctx, next_ctx):
        if ctx is not None:
            s = ctx.sum(1, keepdims=True)
            parts.append(ctx / np.maximum(s, 1))
    return np.concatenate(parts, axis=1)


# ------------------------------------------------------------- gauges
def top_k_accuracy(scores: np.ndarray, sid: np.ndarray,
                   k: int = 1) -> float:
    """scores: (n_sites, n_states) similarity to each state row."""
    top = np.argsort(-scores, axis=1)[:, :k]
    return float((top == sid[:, None]).any(1).mean())


def gallery_decay(scores: np.ndarray, sid: np.ndarray, seed: int,
                  sizes: tuple[int, ...] = (16, 64, 256, 999)
                  ) -> dict[int, float]:
    """Identification accuracy vs gallery size (leak-law Gate 1
    companion: honest identification decays with gallery size; a
    flat curve at 1.0 is the degeneracy signature)."""
    rng = np.random.default_rng(seed)
    n_states = scores.shape[1]
    out = {}
    for g in sizes:
        g = min(g, n_states)
        hits = 0
        for i in range(len(sid)):
            others = rng.choice(n_states - 1, size=g - 1,
                                replace=False)
            others = others + (others >= sid[i])
            gal = np.concatenate(([sid[i]], others))
            hits += int(np.argmax(scores[i, gal]) == 0)
        out[g] = round(hits / max(len(sid), 1), 4)
    return out
