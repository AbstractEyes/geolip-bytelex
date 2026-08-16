"""Matrix pipeline — build lexicons from corpora, project vocabularies.

The productized form of the v1 build (2026-08-16): a CachedProfiler
(contexts repeat massively across a vocabulary, so entropy/PMI lookups
memoize), a lexicon builder over any iterable of byte chunks, and the
full-vocabulary projector that turns a normalized vocab table into a
translation-matrix file.

Vocab table row contract (one JSON object per line):
    {"id": int, "hex": str, "text": str|null, "n_bytes": int,
     "is_special": bool, "continuation": bool}
Matrix row contract:
    {"id": int, "n": int, "hmax": float?, "hargmax": int?,
     "pmin": float?, "word": 0|1}
hmax/hargmax: the token's internal boundary-entropy maximum and its
position — where byte-language says the token divides. pmin: minimum
internal adjacent PMI (cohesion). word: word-gram standing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Iterator

from .core import ByteLexicon, GramSchema


def build_lexicon(chunks: Iterable[bytes],
                  schema: GramSchema | None = None,
                  prune_min_count: int = 2) -> ByteLexicon:
    """Feed any iterable of byte chunks (documents, rows, files) into a
    fresh lexicon. Model-free: chunks are just bytes."""
    lex = ByteLexicon(schema or GramSchema())
    for c in chunks:
        lex.feed(bytes(c))
    if prune_min_count:
        lex.prune(prune_min_count)
    return lex


class CachedProfiler:
    """Memoized profile computation against one lexicon. Contexts and
    byte-pairs repeat across a vocabulary; caching makes full-vocab
    projection (~10^6 tokens) a seconds-scale operation."""

    def __init__(self, lex: ByteLexicon, ctx_n: int = 3):
        self.lex = lex
        self.ctx_n = ctx_n
        self._ent: dict[bytes, float] = {}
        self._pmi: dict[bytes, float] = {}

    def entropy(self, ctx: bytes) -> float:
        if ctx not in self._ent:
            h = self.lex.cond_entropy(ctx)
            self._ent[ctx] = h if h is not None else -1.0
        return self._ent[ctx]

    def pmi(self, pair: bytes) -> float:
        if pair not in self._pmi:
            p = self.lex.pmi(pair[:1], pair[1:])
            self._pmi[pair] = p if p is not None else math.nan
        return self._pmi[pair]

    def profile(self, tok: bytes) -> dict:
        n = self.ctx_n
        out: dict = {"n": len(tok)}
        ents = [self.entropy(tok[i - n:i]) if i >= n else -1.0
                for i in range(1, len(tok))]
        known = [e for e in ents if e >= 0]
        if known:
            mx = max(known)
            out["hmax"] = round(mx, 3)
            out["hargmax"] = ents.index(mx) + 1
        pmis = [p for p in (self.pmi(tok[i - 1:i + 1])
                            for i in range(1, len(tok)))
                if not math.isnan(p)]
        if pmis:
            out["pmin"] = round(min(pmis), 3)
        stripped = bytes(b for b in tok if not self.lex.schema.is_sep(b))
        out["word"] = int(bool(stripped) and stripped in self.lex.words)
        return out


def read_vocab(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def project_vocab(lex: ByteLexicon, vocab_path: str | Path,
                  out_path: str | Path | None = None,
                  split_h_bits: float = 3.0) -> dict:
    """Run one vocabulary through the lexicon -> matrix file + summary.
    Specials are skipped (out-of-alphabet law)."""
    prof = CachedProfiler(lex)
    n = n_word = n_split = n_coh = 0
    h_sum = 0.0
    w = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for r in read_vocab(vocab_path):
            if r.get("is_special"):
                continue
            tok = bytes.fromhex(r["hex"])
            if not tok:
                continue
            p = prof.profile(tok)
            p["id"] = r["id"]
            if w:
                w.write(json.dumps(p) + "\n")
            n += 1
            n_word += p["word"]
            if "hmax" in p:
                h_sum += p["hmax"]
                if p["hmax"] > split_h_bits:
                    n_split += 1
                else:
                    n_coh += 1
    finally:
        if w:
            w.close()
    hn = max(n_split + n_coh, 1)
    return {"tokens": n,
            "word_frac": round(n_word / max(n, 1), 4),
            "divides_frac": round(n_split / hn, 4),
            "cohesive_frac": round(n_coh / hn, 4),
            "mean_internal_hmax_bits": round(h_sum / hn, 4),
            "split_h_bits": split_h_bits}


def alignment_endpoints(lex: ByteLexicon, tok: bytes,
                        split_h_bits: float = 3.0) -> list[int]:
    """Distillation primitive: the byte offsets (within the token) at
    which a consumer should place alignment endpoints. Cohesive tokens
    return [] (align as one span); non-cohesive tokens return their
    internal corpus-entropy boundaries."""
    return lex.split_points(tok, h_quantile_bits=split_h_bits)
