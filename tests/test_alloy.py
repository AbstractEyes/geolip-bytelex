"""alloy tests — synthetic, no downloads, CPU."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geolip.bytelex.alloy import (AlloyLedger, alloy_byte_kl,
                                  alloy_chunk, alloy_frame,
                                  byte_pushforward, descent_gate,
                                  first_byte_map)


def _rows():
    return [{"id": 0, "hex": b" the".hex(), "is_special": False},
            {"id": 1, "hex": b"a".hex(), "is_special": False},
            {"id": 2, "hex": "", "is_special": True},
            {"id": 3, "hex": b" ".hex(), "is_special": False}]


def test_pushforward_exact_and_disclosed():
    cb = first_byte_map(_rows(), 4)
    assert cb[0] == ord("t") and cb[1] == ord("a")
    assert cb[2] == -1 and cb[3] == -1          # special + all-space
    p = torch.tensor([[0.5, 0.3, 0.1, 0.05, 0.05]])   # padded head
    marg, dropped = byte_pushforward(p, cb)
    assert abs(float(dropped) - 0.2) < 1e-6     # special+space+phantom
    assert abs(float(marg[0, ord("t")]) - 0.5 / 0.8) < 1e-6
    assert abs(float(marg.sum()) - 1.0) < 1e-6


def test_byte_kl_zero_at_match_and_t_scaling():
    lg = torch.randn(3, 256)
    tgt = torch.softmax(lg, -1)
    l1 = alloy_byte_kl(lg, tgt, T=1.0)
    assert float(l1.abs().max()) < 1e-5
    lg2 = torch.randn(3, 256)
    a = alloy_byte_kl(lg2, tgt, T=1.0).mean()
    b = alloy_byte_kl(lg2, tgt, T=4.0).mean()
    assert a > 0 and b > 0                       # both live, scaled


def test_chunk_keeps_nm():
    cells = [{"kind": "1x1", "cls": "word", "s_ids": [1], "t_ids": [2],
              "bytes": b" the"},
             {"kind": "nm", "cls": "digit", "s_ids": [3, 4],
              "t_ids": [5], "bytes": b"618"}]
    sl = {0: torch.log_softmax(torch.randn(4), -1),
          1: torch.log_softmax(torch.randn(3), -1)}
    tl = {0: torch.log_softmax(torch.randn(4), -1),
          1: torch.log_softmax(torch.randn(3), -1)}
    k = [0]

    def s_fn(c):
        return sl[k[0]]

    def t_fn(c):
        v = tl[k[0]]
        k[0] += 1
        return v
    out = alloy_chunk(s_fn, t_fn, cells)
    assert float(out["loss_nm"]) != 0.0          # nm ROUTED, not dropped
    assert len(out["rows"]) == 2
    assert out["rows"][1]["cls"] == "digit"


def test_frame_pair_unfused():
    z = torch.randn(8, 16)
    E = torch.randn(5, 16)
    sid = torch.randint(0, 5, (8,))
    out = alloy_frame(z, E[sid], sid, E, torch.tensor(10.0))
    assert out["loss_id"].shape == (8,)
    assert out["loss_mse"].shape == (8,)
    perfect = alloy_frame(E[sid] * 3.0, E[sid], sid, E,
                          torch.tensor(10.0))
    assert float(perfect["loss_mse"].max()) < 1e-6


def test_ledger():
    led = AlloyLedger()
    led.add("byte_kl", "digit", 1.0)
    led.add("byte_kl", "digit", 3.0)
    led.add("frame_id", "word", 0.5)
    r = led.report()
    assert r["byte_kl"]["digit"]["n"] == 2
    assert abs(r["byte_kl"]["digit"]["mean"] - 2.0) < 1e-9


def test_descent_gate_passes_and_fails():
    w = torch.nn.Parameter(torch.randn(4))
    tgt = torch.tensor([1.0, -2.0, 0.5, 3.0])
    g = descent_gate(lambda: ((w - tgt) ** 2).sum(), [w])
    assert g["passed"] and g["rel_drop"] > 0.2
    frozen = torch.nn.Parameter(torch.randn(4))
    try:
        descent_gate(lambda: (tgt ** 2).sum() + 0.0 * frozen.sum(),
                     [frozen])
        raise RuntimeError("gate should have raised")
    except AssertionError:
        pass


if __name__ == "__main__":
    test_pushforward_exact_and_disclosed()
    test_byte_kl_zero_at_match_and_t_scaling()
    test_chunk_keeps_nm()
    test_frame_pair_unfused()
    test_ledger()
    test_descent_gate_passes_and_fails()
    print("alloy tests: 6/6 PASS")
