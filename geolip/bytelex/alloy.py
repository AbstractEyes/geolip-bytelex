"""bytelex.alloy — the byte-mediated distillation loss family.

Every component is separately auditable and returns attribution rows
(never a fused scalar — the BLD trap-gauge). Mismatch cells are
routed, never discarded (the ALM trap-gauge). The pushforward math
grounds in the Byte-Token Representation Lemma (Phan & Ullrich,
arXiv 2410.09303); v1 implements the first-content-byte marginal
form validated by the T-2 battery (canonical-prefix pushforward,
mass loss <=2e-4 measured); the exact multi-byte conditional is an
extension point, not an implicit claim.

Torch enters the library here (losses are training-time objects);
extract/frame stay numpy. Every loss family member is expected to
pass `descent_gate` on oracle targets before a real cell runs — the
loop-control pattern as an affordance: non-descent on oracle means
apparatus bug; non-descent on real targets AFTER an oracle pass
means the objective is inexpressible at these sites. The two are
never again conflated with optimizer failure.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------- pushforward
def first_byte_map(rows: list[dict], n_vocab: int) -> np.ndarray:
    """token id -> first content byte of its expansion (leading
    spaces stripped); -1 = no expansion (special / all-space).
    The T-2-validated gauge form of the pushforward."""
    cb = np.full(n_vocab, -1, dtype=np.int64)
    for r in rows:
        if r["is_special"] or not r["hex"]:
            continue
        e = bytes.fromhex(r["hex"]).lstrip(b" ")
        if e:
            cb[r["id"]] = e[0]
    return cb


def byte_pushforward(probs: torch.Tensor, cb: np.ndarray
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Token distribution(s) -> 256-way first-content-byte marginal.
    probs: (..., V) probabilities (head may be padded past len(cb):
    phantom rows are dropped mass). Returns (marginal, dropped_mass);
    marginal renormalized, dropped mass DISCLOSED, never hidden."""
    v = len(cb)
    over = probs[..., v:].sum(-1)
    p = probs[..., :v]
    keep = torch.tensor(cb >= 0, device=probs.device)
    dropped = over + torch.where(keep, torch.zeros_like(p), p).sum(-1)
    idx = torch.tensor(np.where(cb >= 0, cb, 0), device=probs.device)
    marg = torch.zeros(*p.shape[:-1], 256, device=probs.device,
                       dtype=p.dtype)
    marg.scatter_add_(-1, idx.expand_as(p),
                      torch.where(keep, p, torch.zeros_like(p)))
    s = marg.sum(-1, keepdim=True).clamp_min(1e-12)
    return marg / s, dropped


# ------------------------------------------------------ alloy_byte_kl
def alloy_byte_kl(student_byte_logits: torch.Tensor,
                  teacher_byte_marginal: torch.Tensor,
                  T: float = 1.0) -> torch.Tensor:
    """Tempered byte-KL at an anchor: KL(teacher_T || student_T),
    scaled by T^2 (gradient-magnitude convention). Teacher marginal
    is a probability vector (already pushed forward + renormalized —
    its dropped mass was disclosed at pushforward time, not eaten
    here). Returns per-site loss (no reduction across components)."""
    sl = F.log_softmax(student_byte_logits / T, dim=-1)
    tt = teacher_byte_marginal.clamp_min(1e-12)
    tt = (tt.log() / T).softmax(-1)
    return (T * T) * F.kl_div(sl, tt, reduction="none").sum(-1)


# -------------------------------------------------------- alloy_chunk
def alloy_chunk(student_logprob_fn, teacher_logprob_fn,
                cells: list[dict]) -> dict:
    """Co-boundary chunk matching (the ALM shape) WITHOUT ALM's
    discard: 1x1 cells match token-level logprobs directly; n:m
    cells match BYTE-FACTORIZED within-cell targets (the mismatch is
    routed through composition, and reported in its own row).

    cells: [{"kind": "1x1"|"nm", "cls": str,
             "s_ids": [...], "t_ids": [...],  # per-side token ids
             "bytes": bytes}]                 # the cell's content
    student_logprob_fn(cell) -> per-position student logprobs for
    the cell under its own tokenization; teacher_logprob_fn same.
    Returns {"loss_1x1": tensor, "loss_nm": tensor,
             "rows": [per-cell dicts]} — attribution-ready."""
    l11, lnm, rows = [], [], []
    for c in cells:
        sl = student_logprob_fn(c)
        tl = teacher_logprob_fn(c)
        if c["kind"] == "1x1":
            loss = (tl.exp() * (tl - sl)).sum()
            l11.append(loss)
        else:
            # byte-factorized: both sides expressed on the cell's
            # byte sequence (caller supplies byte-level logprobs for
            # nm cells via its pushforward) — lengths must agree.
            assert sl.shape == tl.shape, "nm cell: byte frames differ"
            loss = (tl.exp() * (tl - sl)).sum()
            lnm.append(loss)
        rows.append({"kind": c["kind"], "cls": c.get("cls", "?"),
                     "loss": float(loss.detach())})
    z = torch.zeros((), dtype=torch.float32)
    return {"loss_1x1": torch.stack(l11).mean() if l11 else z,
            "loss_nm": torch.stack(lnm).mean() if lnm else z,
            "rows": rows}


# -------------------------------------------------------- alloy_frame
def alloy_frame(z: torch.Tensor, target: torch.Tensor,
                sid: torch.Tensor, E_rows: torch.Tensor,
                s: torch.Tensor, b: torch.Tensor | None = None
                ) -> dict:
    """The campaign-A-validated frame pair: identification against
    the codebook (negatives = the inventory, never batch items) +
    unit-sphere MSE to the target vector. z, target: (..., d) frame
    vectors; sid: state ids; E_rows: codebook. Returns both
    components unfused."""
    zn = F.normalize(z, dim=-1)
    lg = s * (zn @ F.normalize(E_rows, dim=-1).T)
    if b is not None:
        lg = lg + b
    lid = F.cross_entropy(lg, sid, reduction="none")
    lmse = ((zn - F.normalize(target, dim=-1)) ** 2).sum(-1)
    return {"loss_id": lid, "loss_mse": lmse}


# --------------------------------------------------------- alloy_loss
def alloy_loss(z: torch.Tensor, sid: torch.Tensor,
               E_rows: torch.Tensor, s: torch.Tensor,
               teacher_targets: dict[str, torch.Tensor],
               b: torch.Tensor | None = None) -> dict:
    """THE ALLOY LOSS (named by Phil, 2026-08-18): multiple teachers
    fused into one student arm through the byte-state frame, each
    component separately assayed. Base metal = identification against
    the codebook (the student's own identity, preserved); each
    teacher contributes a unit-sphere frame regression as an alloying
    element. Returns {"loss_id": ..., "mse_<teacher>": ...} per site
    — components are combined by the CALLER and logged through
    AlloyLedger; this function never fuses them."""
    zn = F.normalize(z, dim=-1)
    lg = s * (zn @ F.normalize(E_rows, dim=-1).T)
    if b is not None:
        lg = lg + b
    out = {"loss_id": F.cross_entropy(lg, sid, reduction="none")}
    for name, tgt in teacher_targets.items():
        out[f"mse_{name}"] = ((zn - F.normalize(tgt, dim=-1)) ** 2
                              ).sum(-1)
    return out


# -------------------------------------------------------- attribution
class AlloyLedger:
    """Per-component, per-class accumulation. The module's contract:
    a training loop logs every component through this, and .report()
    is what ships — a fused scalar is never the record."""

    def __init__(self):
        self._acc: dict[tuple[str, str], list[float]] = {}

    def add(self, component: str, cls: str, value: float):
        self._acc.setdefault((component, cls), []).append(float(value))

    def report(self) -> dict:
        out: dict = {}
        for (comp, cls), vals in sorted(self._acc.items()):
            out.setdefault(comp, {})[cls] = {
                "n": len(vals),
                "mean": round(float(np.mean(vals)), 6)}
        return out


# ------------------------------------------------------- descent gate
def descent_gate(loss_fn, params: list[torch.Tensor],
                 steps: int = 50, lr: float = 1e-2,
                 min_rel_drop: float = 0.2) -> dict:
    """The loop-control pattern as a library affordance: the loss
    must DESCEND on the caller-supplied (oracle-target) closure
    before any real cell runs. Pure Adam wd=0 (house law). Returns
    {"passed", "first", "last", "rel_drop"}; raises on failure so a
    campaign cannot silently proceed with an apparatus bug."""
    opt = torch.optim.Adam(params, lr=lr, weight_decay=0.0)
    first = last = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        if first is None:
            first = float(loss.detach())
        loss.backward()
        opt.step()
        last = float(loss.detach())
    rel = (first - last) / max(abs(first), 1e-9)
    out = {"passed": rel >= min_rel_drop, "first": round(first, 6),
           "last": round(last, 6), "rel_drop": round(rel, 4)}
    if not out["passed"]:
        raise AssertionError(f"descent gate FAILED: {out} — apparatus "
                             f"bug (oracle targets must descend)")
    return out
