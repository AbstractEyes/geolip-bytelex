"""Core validations: hand-computed information quantities, persistence,
alphabet parametricity. Deterministic — no RNG anywhere."""
import math

import pytest

from geolip.bytelex import ByteLexicon, GramSchema


def lex_of(chunks, schema=None):
    lex = ByteLexicon(schema or GramSchema(hash_ns=(9,)))
    for c in chunks:
        lex.feed(c)
    return lex


# ------------------------------------------------------------- counting
def test_gram_counts_exact():
    lex = lex_of([b"abab"])
    assert lex.counts[1][b"a"] == 2 and lex.counts[1][b"b"] == 2
    assert lex.counts[2][b"ab"] == 2 and lex.counts[2][b"ba"] == 1
    assert lex.counts[3][b"aba"] == 1 and lex.counts[3][b"bab"] == 1
    assert lex.n_bytes == 4


def test_word_grams_and_separators():
    lex = lex_of([b"sun fog sun"])
    assert lex.words[b"sun"] == 2 and lex.words[b"fog"] == 1


# ---------------------------------------------------- entropy (by hand)
def test_cond_entropy_exact_one_bit():
    # ctx "abc" occurs 4x, followed by X,Y,X,Y -> H = 1.0 bit exactly
    lex = lex_of([b"abcXabcYabcXabcY"])
    assert lex.cond_entropy(b"abc") == pytest.approx(1.0)


def test_cond_entropy_zero_when_deterministic():
    lex = lex_of([b"q" * 32])
    assert lex.cond_entropy(b"qqq") == pytest.approx(0.0)


def test_cond_entropy_refuses_thin_evidence():
    lex = lex_of([b"abcX"])          # ctx count 1 < 4
    assert lex.cond_entropy(b"abc") is None


# ----------------------------------------------------------------- PMI
def test_pmi_positive_for_bound_pair():
    lex = lex_of([b"aa qu aa qu aa qu aa qu "])
    p = lex.pmi(b"q", b"u")
    assert p is not None and p > 1.0        # q and u always adjacent


def test_pmi_none_for_unseen_pair():
    lex = lex_of([b"aaaa bbbb"])
    assert lex.pmi(b"a", b"b") is None or lex.pmi(b"b", b"a") is None


# --------------------------------------------------------- split points
def _branchy_corpus():
    # ctx "abc" fans out to 10 distinct successors -> H ~ log2(10) > 3
    body = b"".join(b"abc" + bytes([c]) + b" " for c in b"defghijklm")
    return [body * 2]


def test_split_points_open_at_high_entropy():
    lex = lex_of(_branchy_corpus())
    assert lex.split_points(b"abcdef") == [3]


def test_split_points_empty_for_cohesive():
    lex = lex_of([b"q" * 64])
    assert lex.split_points(b"qqqqq") == []


# -------------------------------------------------------------- profile
def test_profile_fields_and_word_standing():
    lex = lex_of([b"sun fog sun fog sun fog sun fog "])
    p = lex.profile(b"sun")
    assert p["is_word"] and p["word_count"] == 4
    p2 = lex.profile(b" sun")           # separator stripped for standing
    assert p2["is_word"]


# ------------------------------------------------- alphabet parametric
def test_small_alphabet():
    schema = GramSchema(alphabet_size=4, char_ns=(1, 2, 3, 4),
                        hash_ns=(), seps=bytes([3]))
    lex = ByteLexicon(schema)
    lex.feed(bytes([0, 1, 2, 3] * 8))
    h = lex.cond_entropy(bytes([0, 1, 2]))
    assert h == pytest.approx(0.0)      # 012 always followed by 3
    assert lex.words[bytes([0, 1, 2])] == 8


# ------------------------------------------------------------- persist
def test_save_load_roundtrip(tmp_path):
    lex = lex_of([b"abcXabcYabcXabcY sun fog sun fog"])
    lex.save(tmp_path / "lex")
    back = ByteLexicon.load(tmp_path / "lex")
    assert back.n_bytes == lex.n_bytes
    assert back.counts[3] == lex.counts[3]
    assert back.words == lex.words
    assert back.cond_entropy(b"abc") == pytest.approx(
        lex.cond_entropy(b"abc"))
    assert back.schema.to_dict() == lex.schema.to_dict()


def test_prune_drops_hapax():
    lex = lex_of([b"abab zz"])
    assert lex.counts[2][b"zz"] == 1
    lex.prune(min_count=2)
    assert b"zz" not in lex.counts[2]
    assert lex.counts[2][b"ab"] == 2
