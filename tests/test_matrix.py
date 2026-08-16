"""Pipeline validations: builder, cached profiler equivalence,
full-vocab projection contract, distillation endpoints."""
import json

import pytest

from geolip.bytelex import (ByteLexicon, CachedProfiler, GramSchema,
                            alignment_endpoints, build_lexicon,
                            project_vocab)

CORPUS = [b"".join(b"abc" + bytes([c]) + b" " for c in b"defghijklm") * 2,
          b"sun fog sun fog sun fog sun fog ",
          b"abcXabcYabcXabcY " * 3]


def test_build_lexicon_from_iterable():
    lex = build_lexicon(CORPUS, prune_min_count=0)
    assert lex.n_bytes == sum(len(c) for c in CORPUS)
    assert lex.words[b"sun"] == 4


def test_cached_profiler_matches_core():
    lex = build_lexicon(CORPUS, prune_min_count=0)
    prof = CachedProfiler(lex)
    for tok in (b"abcdef", b"sun", b"abcX", b"qq"):
        fast, slow = prof.profile(tok), lex.profile(tok)
        if "hmax" in fast:
            assert fast["hmax"] == pytest.approx(
                round(slow["internal_H_max"], 3))
            assert fast["hargmax"] == slow["internal_H_argmax"]
        assert fast["word"] == int(slow["is_word"])
    # cache actually populated and reused
    assert prof._ent and prof.profile(b"abcdef") == prof.profile(b"abcdef")


def _write_vocab(path):
    rows = [
        {"id": 0, "hex": b"sun".hex(), "text": "sun", "n_bytes": 3,
         "is_special": False, "continuation": False},
        {"id": 1, "hex": b"abcdef".hex(), "text": "abcdef", "n_bytes": 6,
         "is_special": False, "continuation": False},
        {"id": 2, "hex": b"<|special|>".hex(), "text": "<|special|>",
         "n_bytes": 11, "is_special": True, "continuation": False},
        {"id": 3, "hex": "", "text": "", "n_bytes": 0,
         "is_special": False, "continuation": False},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_project_vocab_contract(tmp_path):
    lex = build_lexicon(CORPUS, prune_min_count=0)
    vp, mp = tmp_path / "vocab_t.jsonl", tmp_path / "matrix_t.jsonl"
    _write_vocab(vp)
    summary = project_vocab(lex, vp, mp, split_h_bits=3.0)
    # special + empty rows skipped -> 2 projected tokens
    assert summary["tokens"] == 2
    rows = [json.loads(l) for l in open(mp, encoding="utf-8")]
    assert {r["id"] for r in rows} == {0, 1}
    by_id = {r["id"]: r for r in rows}
    assert by_id[0]["word"] == 1                      # "sun" is a word
    assert by_id[1]["hargmax"] == 3                   # divides after "abc"
    assert summary["divides_frac"] > 0
    assert summary["cohesive_frac"] + summary["divides_frac"] == \
        pytest.approx(1.0)


def test_alignment_endpoints_primitive():
    lex = build_lexicon(CORPUS, prune_min_count=0)
    assert alignment_endpoints(lex, b"abcdef") == [3]   # split the stitch
    assert alignment_endpoints(lex, b"sun") == []       # cohesive: one span


def test_schema_modularity_extra_view():
    # adding a gram view is config, not code
    schema = GramSchema(char_ns=(1, 2, 3, 4), hash_ns=(6, 9))
    lex = build_lexicon(CORPUS, schema=schema, prune_min_count=0)
    assert 6 in lex.hcounts and sum(lex.hcounts[6].values()) > 0
