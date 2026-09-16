"""Trigram atlas — a deterministic coordinate space over byte trigrams, weighted by token numerics.

The space is complete a priori: every 3-byte sequence is one cell, indexed b0<<16 | b1<<8 | b2
(16,777,216 possible). A token's byte string is a PATH of overlapping cells; each step also defines
an EDGE (cell -> next byte), and a token's final cell receives END MASS. Cells no token touches are
dead heads: they never enter the dictionaries, so strength queries return 0 through dict-default —
ignored at zero cost.

Weights come from tokenizer vocabularies alone (no corpus anywhere): BPE token ids are merge-ordered,
so 1/log2(2 + id) is a frequency-rank proxy ("rank" scheme); "ordinal" is the linear variant; "flat"
weights every token 1.0, and summing flat fields across many tokenizers gives the consensus field —
how much of the fleet allocates vocabulary to a cell.

Per-cell caveats, from the same numerics:
    entropy   H(next byte | cell) in bits over within-token continuations
    top       the dominant continuation's share (a high value marks a forced-continuation trap)
    close     end_mass / (end_mass + continuation mass) — can a unit plausibly stop here?

Validation on record: the fleet-consensus `top` caveat reproduced a trained byte model's per-word
accuracy ladder at Spearman -0.976 (the corpus boundary functional scored -0.62 on the same words),
and every scheme identified the three worst words exactly. Ledger: huggingface.co/AbstractPhil/
geolip-bytelex, btx_e003/atlas_proto_ledger.json.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

LETTERS = frozenset(range(97, 123))


def cell_index(b0: int, b1: int, b2: int) -> int:
    return (b0 << 16) | (b1 << 8) | b2


def cell_bytes(c: int) -> bytes:
    return bytes(((c >> 16) & 255, (c >> 8) & 255, c & 255))


def load_vocab_rows(path: str | Path, min_bytes: int = 3):
    """vocab_<name>.jsonl (the fleet-extraction format) -> [(id, token_bytes)], specials skipped."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("is_special"):
                continue
            b = bytes.fromhex(r["hex"])
            if len(b) >= min_bytes:
                rows.append((r["id"], b))
    return rows


def load_vocab_words(path: str | Path):
    """The vocabulary's word surface: token texts stripped of space markers, lowercased, alphabetic
    only. Used as the no-corpus novelty screen for minting."""
    words = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            t = (r.get("text") or "").lstrip(" Ġ▁").strip().lower()
            if t.isalpha():
                words.add(t)
    return words


class WeightField:
    """One weight field over the atlas. strength/edges/end_mass hold only live cells —
    absence IS the dead-head state, and queries fall through to 0."""

    def __init__(self):
        self.strength: dict[int, float] = {}
        self.edges: dict[int, dict[int, float]] = {}
        self.end_mass: dict[int, float] = {}

    # ------------------------------------------------------------- build
    def add_token(self, b: bytes, w: float):
        n = len(b)
        for i in range(n - 2):
            c = cell_index(b[i], b[i + 1], b[i + 2])
            self.strength[c] = self.strength.get(c, 0.0) + w
            if i + 3 < n:
                d = self.edges.setdefault(c, {})
                d[b[i + 3]] = d.get(b[i + 3], 0.0) + w
            else:
                self.end_mass[c] = self.end_mass.get(c, 0.0) + w

    @classmethod
    def from_rows(cls, rows, scheme: str = "rank") -> "WeightField":
        f = cls()
        V = max((i for i, _ in rows), default=0) + 1
        for tid, b in rows:
            if scheme == "rank":
                w = 1.0 / math.log2(2 + tid)
            elif scheme == "ordinal":
                w = (V - tid) / V
            elif scheme == "flat":
                w = 1.0
            else:
                raise ValueError(f"unknown scheme {scheme!r}")
            f.add_token(b, w)
        return f

    @classmethod
    def consensus(cls, fields: list["WeightField"]) -> "WeightField":
        """Sum of fields. Feed it flat fields (one per tokenizer) for the consensus form."""
        out = cls()
        for f in fields:
            for c, v in f.strength.items():
                out.strength[c] = out.strength.get(c, 0.0) + v
            for c, d in f.edges.items():
                o = out.edges.setdefault(c, {})
                for b, v in d.items():
                    o[b] = o.get(b, 0.0) + v
            for c, v in f.end_mass.items():
                out.end_mass[c] = out.end_mass.get(c, 0.0) + v
        return out

    # ------------------------------------------------------------- query
    def cell_strength(self, c: int) -> float:
        return self.strength.get(c, 0.0)

    def caveat(self, c: int):
        """-> (entropy_bits, top_share, closability) for a live cell, None for a dead head."""
        e = self.edges.get(c, {})
        cont = sum(e.values())
        endm = self.end_mass.get(c, 0.0)
        if cont + endm <= 0:
            return None
        h = 0.0
        top = 0.0
        for v in e.values():
            p = v / cont
            h -= p * math.log2(p)
            top = max(top, p)
        return (h, top, endm / (endm + cont))

    def live_cells(self):
        return set(self.strength)

    def coverage_gap(self, other: "WeightField"):
        """Cells live here and dead there — e.g. what a teacher's field knows in space the
        student's field never witnessed."""
        return self.live_cells() - other.live_cells()

    def spectrum(self, b: bytes):
        """A byte string's alignment spectrum: per overlapping step, the cell, its strength,
        and its caveat triple (None on dead heads)."""
        out = []
        for i in range(len(b) - 2):
            c = cell_index(b[i], b[i + 1], b[i + 2])
            out.append({"cell": c, "strength": self.strength.get(c, 0.0), "caveat": self.caveat(c)})
        return out

    # ------------------------------------------------------------- minting
    def word_end_caveat(self, word: bytes):
        return self.caveat(cell_index(word[-3], word[-2], word[-1]))

    def mint_lexicon_program(self, *, n_easy=0, n_trap=0, pair_prefixes=(), seed=0,
                             length_range=(4, 6), easy_band=(0.0, 0.42), trap_band=(0.55, 1.0),
                             known_words: set | None = None, avoid: set | None = None,
                             cross_cap: int = 1, oversample: int = 4):
        """Mint a GRADED lexicon program: an easy-close band, a trap band, and optional
        shared-prefix pairs — difficulty as a measured dial rather than an accident.

        Bands select on the word-end dominant-continuation share (`word_end_caveat` top):
        easy words sit where many continuations stay plausible, trap words where one
        continuation dominates (the field "refuses to close" them). `pair_prefixes` mints
        one word pair per prefix (first half of the list in the easy band, second half in
        the trap band), sharing that onset by design — training material for commitment
        between look-alike surfaces. Screens: novelty against `known_words` and `avoid`,
        and pairwise flanked-trigram sharing capped at `cross_cap` across all minted words
        (within-pair prefixes exempt by construction). Deterministic under `seed`.

        Returns {"easy": [...], "trap": [...], "pairs": [[a, b], ...]}. Raises ValueError
        when the field cannot fill a band — widen the band, shorten the words, or feed a
        richer field. Validation on record: trap-band-trained models flatten a measured
        per-word failure ladder with transfer (huggingface.co/AbstractPhil/geolip-bytelex,
        btx_e003 artifacts)."""
        known = known_words or set()
        avoid = set(a.lower() for a in (avoid or set()))
        taken: list[set] = []
        out = {"easy": [], "trap": [], "pairs": []}

        def flanked(w: str):
            s = " " + w + " "
            return {s[i:i + 3] for i in range(len(s) - 2)}

        def cross_filter(cands, want):
            picked = []
            for w in cands:
                t = flanked(w)
                if all(len(t & tt) <= cross_cap for tt in taken):
                    picked.append(w); taken.append(t)
                    if len(picked) == want:
                        break
            return picked

        for name, n, band in (("easy", n_easy, easy_band), ("trap", n_trap, trap_band)):
            if not n:
                continue
            cands = self.mint_words(n * oversample, seed=seed + (0 if name == "easy" else 1),
                                    length_range=length_range, top_band=band, known_words=known,
                                    avoid=avoid | set(out["easy"]))
            got = cross_filter(cands, n)
            if len(got) < n:
                raise ValueError(f"{name} band yielded {len(got)}/{n}; widen the band or the field")
            out[name] = got
        for k, pf in enumerate(pair_prefixes):
            band = easy_band if 2 * k < len(pair_prefixes) else trap_band
            got = self.mint_words(2 * oversample, seed=seed + 100 + k, length_range=length_range,
                                  top_band=band, known_words=known,
                                  avoid=avoid | set(out["easy"]) | set(out["trap"])
                                  | {w for p in out["pairs"] for w in p},
                                  max_shared_trigrams=3, prefix=pf)
            pick = []
            for w in got:
                t = flanked(w)
                if all(len(t & tt) <= cross_cap for tt in taken) and w not in pick:
                    pick.append(w)
                    if len(pick) == 2:
                        break
            if len(pick) < 2:
                raise ValueError(f"prefix {pf!r} yielded {len(pick)}/2; widen the band or the field")
            for w in pick:
                taken.append(flanked(w))
            out["pairs"].append(pick)
        return out

    def mint_words(self, n: int, seed: int, length_range=(4, 6), top_band=(0.0, 1.0),
                   known_words: set | None = None, avoid: set | None = None,
                   max_shared_trigrams: int = 1, prefix: bytes | None = None,
                   max_tries: int = 200_000):
        """Mint n novel lowercase words by walking live edges, weighted by the field.

        Screens, all from token numerics: every internal transition rides a live edge (spellable);
        the word appears in no known vocabulary surface (known_words) and no caller blocklist
        (avoid); the word-end `top` caveat sits inside top_band (the trap dial); pairwise, minted
        words share at most max_shared_trigrams space-flanked trigrams (distinct identities in a
        trigram-composed input space). `prefix` pins the first bytes for shared-onset groups."""
        rng = random.Random(seed)
        known = known_words or set()
        avoid = {a.lower() for a in (avoid or set())}
        starts = [c for c in self.strength
                  if all(x in LETTERS for x in cell_bytes(c)) and (prefix is None or cell_bytes(c).startswith(prefix))]
        if not starts:
            return []
        sw = [self.strength[c] for c in starts]

        def flanked(word: bytes):
            s = b" " + word + b" "
            return {s[i:i + 3] for i in range(len(s) - 2)}

        out, out_tris = [], []
        for _ in range(max_tries):
            if len(out) >= n:
                break
            L = rng.randint(*length_range)
            c = rng.choices(starts, weights=sw)[0]
            word = bytearray(cell_bytes(c))
            ok = True
            while len(word) < L:
                e = {b: v for b, v in self.edges.get(c, {}).items() if b in LETTERS}
                if not e:
                    ok = False
                    break
                nb = rng.choices(list(e), weights=list(e.values()))[0]
                word.append(nb)
                c = cell_index(word[-3], word[-2], word[-1])
            if not ok:
                continue
            w = bytes(word)
            txt = w.decode()
            if txt in known or txt in avoid or any(txt == o.decode() for o in out):
                continue
            cv = self.word_end_caveat(w)
            if cv is None or not (top_band[0] <= cv[1] <= top_band[1]):
                continue
            tris = flanked(w)
            if any(len(tris & t) > max_shared_trigrams for t in out_tris):
                continue
            out.append(w)
            out_tris.append(tris)
        return [w.decode() for w in out]
