"""Token -> byte-expansion table extraction (the fleet extractor).

Turns any tokenizer into the vocab_<name>.jsonl table the rest of
bytelex consumes: one row per token id, `hex` = the token's exact byte
expansion, specials flagged and given no expansion (the out-of-alphabet
law: control symbols are a second alphabet, never byte content).

Two families are supported:
- gpt2-byte-unicode: byte-level BPE storing bytes as remapped unicode
  chars (GPT-2, Qwen, Llama-3, DeepSeek, o200k-style vocabularies).
- sentencepiece: '▁' word-boundary convention + <0xNN> byte-
  fallback rows (Llama-2, T5-class).

The extractor is pure logic over a duck-typed tokenizer (needs
convert_ids_to_tokens, __len__, all_special_tokens, encode/__call__);
transformers is never imported here. Every extraction ends with a
ROUND-TRIP GATE: a probe sentence must reassemble byte-exactly from
the table or the extraction raises.
"""
from __future__ import annotations

import json
from pathlib import Path

_PROBE = "The answer is 367 — naturally."
# Lossy families gate on ASCII: representability of exotic bytes is
# coverage()'s job, the gate only checks reassembly of what the vocab
# CAN say (t5-unigram has no em dash; that is disclosure, not error).
_PROBE_ASCII = "The answer is 367, naturally."


def _gpt2_unicode_to_byte() -> dict[str, int]:
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs)}


def _specials(tokenizer) -> set[str]:
    sp = set(getattr(tokenizer, "all_special_tokens", []) or [])
    extra = getattr(tokenizer, "additional_special_tokens", None)
    if extra:
        sp |= set(extra)
    return sp


def _expand_gpt2(token: str, u2b: dict[str, int]) -> bytes | None:
    try:
        return bytes(u2b[c] for c in token)
    except KeyError:
        return None


def _expand_sentencepiece(token: str) -> bytes | None:
    if (len(token) == 6 and token.startswith("<0x")
            and token.endswith(">")):
        try:
            return bytes([int(token[3:5], 16)])
        except ValueError:
            return None
    return token.replace("▁", " ").encode("utf-8")


def _expand_wordpiece(token: str) -> bytes | None:
    """BERT wordpiece: '##x' continues the current word (bytes 'x');
    a bare token starts a word (bytes ' ' + token). LOSSY by design
    for uncased vocabularies (case folded) and around punctuation
    (whitespace canonicalized) — gate in 'normalized' mode and record
    the lossiness; never pretend byte-exactness."""
    if token.startswith("[") and token.endswith("]"):
        return None                      # [CLS]/[SEP]/[unused..] etc.
    if token.startswith("##"):
        return token[2:].encode("utf-8")
    return (" " + token).encode("utf-8")


def detect_family(tokenizer) -> str:
    """gpt2 / sentencepiece / wordpiece, by convention signature:
    'Ġ' marks gpt2 space, '▁' marks sentencepiece, '##' marks
    wordpiece continuations. Ascii merges and <0xNN> fallback rows are
    expandable under multiple maps, so markers decide, coverage is the
    last resort."""
    u2b = _gpt2_unicode_to_byte()
    sp = _specials(tokenizer)
    seen = hit = g_space = sp_space = wp_cont = 0
    for tid in range(min(len(tokenizer), 200_000)):
        t = tokenizer.convert_ids_to_tokens(tid)
        if t is None or t in sp or not isinstance(t, str):
            continue
        seen += 1
        g_space += "Ġ" in t                     # gpt2 space marker
        sp_space += "▁" in t                    # sentencepiece marker
        wp_cont += t.startswith("##")           # wordpiece continuation
        if _expand_gpt2(t, u2b) is not None:
            hit += 1
    if seen == 0:
        raise ValueError("tokenizer yielded no ordinary tokens to probe")
    best = max(("gpt2", g_space), ("sentencepiece", sp_space),
               ("wordpiece", wp_cont), key=lambda kv: kv[1])
    if best[1] > 0:
        return best[0]
    return "gpt2" if hit / seen > 0.95 else "sentencepiece"


def token_byte_table(tokenizer, family: str | None = None) -> list[dict]:
    """Full matrix.py row contract: {id, hex, text, n_bytes,
    is_special, continuation}. The continuation flag separates the
    boundary alphabet from content (out-of-alphabet law): a
    word-START expansion carries the leading separator byte; a
    CONTINUATION expansion is pure content."""
    family = family or detect_family(tokenizer)
    u2b = _gpt2_unicode_to_byte()
    sp = _specials(tokenizer)
    rows = []
    for tid in range(len(tokenizer)):
        t = tokenizer.convert_ids_to_tokens(tid)
        special = (t is None or t in sp
                   or (isinstance(t, str) and t.startswith("<|")
                       and t.endswith("|>")))
        exp = None
        cont = False
        if not special:
            if family == "gpt2":
                exp = _expand_gpt2(t, u2b)
            elif family == "wordpiece":
                exp = _expand_wordpiece(t)
                cont = t.startswith("##")
            else:
                exp = _expand_sentencepiece(t)
                cont = not t.startswith("▁") and not (
                    len(t) == 6 and t.startswith("<0x"))
            if family == "gpt2" and exp is not None:
                cont = not exp[:1].isspace()
            special = exp is None
        txt = exp.decode("utf-8", errors="replace") if exp else None
        rows.append({"id": tid, "hex": exp.hex() if exp else "",
                     "text": txt, "n_bytes": len(exp) if exp else 0,
                     "is_special": bool(special),
                     "continuation": bool(cont and not special)})
    return rows


def _encode_ids(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _normalize(s: str) -> str:
    """Gate-compare form for lossy families: case folded, ALL
    whitespace removed. Wordpiece legally re-spaces punctuation
    ('naturally .' for 'naturally.'), so single-space collapse is
    still too strict; any CONTENT drop still fails."""
    return "".join(s.lower().split())


def roundtrip_gate(tokenizer, rows: list[dict], probe: str = _PROBE,
                   mode: str = "exact") -> None:
    """The extraction is not trusted until a probe sentence reassembles
    from the table. mode='exact': byte-exact (gpt2-class, byte-complete
    vocabularies). mode='normalized': case-folded and whitespace-
    canonicalized comparison (sentencepiece/wordpiece — LOSSY
    tokenizers may legally drop case and re-space punctuation, and
    that lossiness is disclosed, never hidden behind a weaker gate
    silently)."""
    exp = {r["id"]: bytes.fromhex(r["hex"]) for r in rows
           if not r["is_special"] and r["hex"]}
    ids = _encode_ids(tokenizer, probe)
    joined = b"".join(exp.get(i, b"") for i in ids)
    got = joined.decode("utf-8", errors="replace")
    ok = (got == probe if mode == "exact"
          else _normalize(got) == _normalize(probe))
    if not ok:
        raise AssertionError(
            f"round-trip gate FAILED ({mode}): {got!r} != {probe!r}")


def coverage(rows: list[dict], corpus: bytes) -> dict:
    """Fraction of corpus bytes representable by the table — the
    translation-loss floor for non-byte-complete vocabularies (t5
    unigram has no byte fallback; wordpiece drops case). Greedy
    longest-match walk over a byte trie of the expansions."""
    root: dict = {}
    for r in rows:
        if r["is_special"] or not r["hex"]:
            continue
        node = root
        for b in bytes.fromhex(r["hex"]):
            node = node.setdefault(b, {})
        node[-1] = True                      # terminal
    covered = i = 0
    n = len(corpus)
    while i < n:
        node, j, last = root, i, -1
        while j < n and corpus[j] in node:
            node = node[corpus[j]]
            j += 1
            if -1 in node:
                last = j
        if last > i:
            covered += last - i
            i = last
        else:
            i += 1
    return {"bytes": n, "covered": covered,
            "coverage": covered / max(n, 1)}


def write_vocab_jsonl(tokenizer, path: str | Path,
                      family: str | None = None,
                      probe: str = _PROBE) -> dict:
    """Extract, gate, write. Returns a summary dict. Gate mode follows
    the family: gpt2 is byte-exact; sentencepiece/wordpiece gate in
    normalized mode and the summary carries lossy=True so no consumer
    mistakes the table for byte-complete."""
    family = family or detect_family(tokenizer)
    rows = token_byte_table(tokenizer, family)
    mode = "exact" if family == "gpt2" else "normalized"
    if mode == "normalized" and probe == _PROBE:
        probe = _PROBE_ASCII
    roundtrip_gate(tokenizer, rows, probe, mode=mode)
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_sp = sum(r["is_special"] for r in rows)
    return {"family": family, "n_tokens": len(rows),
            "n_special": n_sp, "lossy": family != "gpt2",
            "gate_mode": mode, "path": str(path)}
