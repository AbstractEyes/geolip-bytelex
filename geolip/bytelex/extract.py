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


def detect_family(tokenizer) -> str:
    """'gpt2' when the byte-unicode map covers ordinary tokens,
    else 'sentencepiece'."""
    u2b = _gpt2_unicode_to_byte()
    sp = _specials(tokenizer)
    seen = hit = g_space = sp_space = 0
    for tid in range(min(len(tokenizer), 4000)):
        t = tokenizer.convert_ids_to_tokens(tid)
        if t is None or t in sp or not isinstance(t, str):
            continue
        seen += 1
        g_space += "Ġ" in t                     # gpt2 space marker
        sp_space += "▁" in t                    # sentencepiece marker
        if _expand_gpt2(t, u2b) is not None:
            hit += 1
    if seen == 0:
        raise ValueError("tokenizer yielded no ordinary tokens to probe")
    # the space-marker convention is the decisive signature: ascii
    # merges and <0xNN> fallback rows are expandable under BOTH maps
    if g_space or sp_space:
        return "gpt2" if g_space >= sp_space else "sentencepiece"
    return "gpt2" if hit / seen > 0.95 else "sentencepiece"


def token_byte_table(tokenizer, family: str | None = None) -> list[dict]:
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
        if not special:
            exp = (_expand_gpt2(t, u2b) if family == "gpt2"
                   else _expand_sentencepiece(t))
            special = exp is None
        rows.append({"id": tid, "hex": exp.hex() if exp else "",
                     "is_special": bool(special)})
    return rows


def _encode_ids(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def roundtrip_gate(tokenizer, rows: list[dict],
                   probe: str = _PROBE) -> None:
    """The extraction is not trusted until a probe sentence reassembles
    BYTE-EXACTLY from the table."""
    exp = {r["id"]: bytes.fromhex(r["hex"]) for r in rows
           if not r["is_special"] and r["hex"]}
    ids = _encode_ids(tokenizer, probe)
    joined = b"".join(exp.get(i, b"") for i in ids)
    if joined.decode("utf-8", errors="replace") != probe:
        raise AssertionError(
            f"round-trip gate FAILED: {joined!r} != {probe!r}")


def write_vocab_jsonl(tokenizer, path: str | Path,
                      family: str | None = None,
                      probe: str = _PROBE) -> dict:
    """Extract, gate, write. Returns a summary dict."""
    family = family or detect_family(tokenizer)
    rows = token_byte_table(tokenizer, family)
    roundtrip_gate(tokenizer, rows, probe)
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_sp = sum(r["is_special"] for r in rows)
    return {"family": family, "n_tokens": len(rows),
            "n_special": n_sp, "path": str(path)}
