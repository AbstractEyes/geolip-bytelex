"""Frame-math tests: whitening, procrustes, splits, gauges."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geolip.bytelex.frame import (apply_whitening, fit_whitening,
                                  gallery_decay, procrustes,
                                  split_sites, top_k_accuracy)


def test_whitening_identity_cov():
    rng = np.random.default_rng(0)
    m = np.eye(16) + 0.3 * rng.normal(size=(16, 16))
    a = rng.normal(size=(4000, 16)) @ m + 5.0
    mu, w = fit_whitening(a, eps=1e-6)
    z = apply_whitening(a, mu, w)
    c = np.cov(z, rowvar=False)
    assert np.abs(c - np.eye(16)).max() < 0.05
    assert np.abs(z.mean(0)).max() < 0.05


def test_procrustes_recovers_rotation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 12)).astype(np.float64)
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    r = procrustes(x, x @ q)
    assert np.abs(r - q).max() < 1e-8
    assert np.abs(r.T @ r - np.eye(12)).max() < 1e-10


def test_procrustes_rectangular():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 16))
    y = rng.normal(size=(300, 8))
    r = procrustes(x, y)
    assert r.shape == (16, 8)
    assert np.abs(r.T @ r - np.eye(8)).max() < 1e-8


def test_split_sites_stratified_disjoint():
    rng = np.random.default_rng(3)
    sid = np.repeat(np.arange(20), 10)
    rng.shuffle(sid)
    sp = split_sites(sid, seed=7)
    all_ix = np.concatenate([sp["fit"], sp["train"], sp["eval"]])
    assert len(all_ix) == len(sid)
    assert len(np.unique(all_ix)) == len(sid)
    for s in range(20):                     # every state in every split
        assert (sid[sp["fit"]] == s).any()
        assert (sid[sp["eval"]] == s).any()


def test_split_sites_count2_one_one():
    sid = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2])
    sp = split_sites(sid, seed=0)
    assert not (sid[sp["fit"]] == 1).any()       # rare never feeds fit
    assert (sid[sp["train"]] == 1).sum() == 1    # count-2: 1 train
    assert (sid[sp["eval"]] == 1).sum() == 1     # count-2: 1 eval
    assert (sid[sp["train"]] == 2).sum() == 1    # count-1: train only
    assert not (sid[sp["eval"]] == 2).any()


def test_state_byte_features_separate_states():
    from geolip.bytelex.frame import state_byte_features
    texts = [b"the", b"The", b"367", b"368", b"and"]
    f = state_byte_features(texts)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            assert np.abs(f[i] - f[j]).sum() > 1e-6   # all distinct
    assert f.shape[0] == 5 and f.shape[1] == 513


def test_gauges():
    n, m = 200, 30
    rng = np.random.default_rng(4)
    sid = rng.integers(0, m, size=n)
    perfect = np.zeros((n, m))
    perfect[np.arange(n), sid] = 1.0
    assert top_k_accuracy(perfect, sid, 1) == 1.0
    noise = rng.normal(size=(n, m))
    assert top_k_accuracy(noise, sid, 1) < 0.2
    d = gallery_decay(noise, sid, seed=0, sizes=(4, 16))
    assert d[4] > d[16]                      # honest decay


if __name__ == "__main__":
    test_whitening_identity_cov()
    test_procrustes_recovers_rotation()
    test_procrustes_rectangular()
    test_split_sites_stratified_disjoint()
    test_split_sites_count2_one_one()
    test_state_byte_features_separate_states()
    test_gauges()
    print("frame tests: 6/6 PASS")
